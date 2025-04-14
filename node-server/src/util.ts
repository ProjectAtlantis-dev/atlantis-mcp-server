import { existsSync, readFileSync, unlinkSync } from 'fs';
import process from 'process'; // Explicit import for process
import winston from 'winston'; // Import winston type for the logger parameter
import io from 'socket.io-client'; // Import for cloud connection
import os from 'os'; // Import for hostname access

// Import shared state functions
import { 
    getCloudSocket,
    setCloudSocket,
    getCloudReconnectTimer, 
    setCloudReconnectTimer, 
    getCloudConnectionAttempts,
    setCloudConnectionAttempts,
    resetCloudConnectionAttempts,
    incrementCloudConnectionAttempts,
    getLogger
} from './state';

// --- Function to Check Existing Process --- //
export function checkAndHandleExistingProcess(customLogger: winston.Logger, PID_FILE: string): void {
    // Use the passed logger for this function instead of global
    const logger = customLogger;
    logger.debug(`Checking for existing PID file: ${PID_FILE}`);
    if (existsSync(PID_FILE)) {
        logger.info(`Found PID file: ${PID_FILE}`);
        let pid: number;
        try {
            const pidString = readFileSync(PID_FILE, 'utf8');
            pid = parseInt(pidString.trim(), 10);
            if (isNaN(pid)) {
                logger.warn(`PID file (${PID_FILE}) contains invalid content: '${pidString}'. Removing stale file.`);
                unlinkSync(PID_FILE);
                return; // Continue startup
            }
            logger.info(`PID read from file: ${pid}`);
        } catch (readError: any) {
            logger.error(`Error reading PID file (${PID_FILE}): ${readError.message}. Removing potentially corrupt file.`);
            try { unlinkSync(PID_FILE); } catch (unlinkErr: any) { /* Ignore inner error */ }
            return; // Continue startup, although PID file was problematic
        }

        try {
            // Check if process exists by sending signal 0
            process.kill(pid, 0);
            // If kill succeeds without error, the process exists
            logger.info(`ℹ️ Node server is already running with PID: ${pid}. Exiting...`);
            process.exit(0); // Exit gracefully
        } catch (err: any) {
            if (err.code === 'ESRCH') {
                // Process doesn't exist (Error: Search) - stale PID file
                logger.warn(`🧹 Process with PID ${pid} not found. Removing stale PID file: ${PID_FILE}`);
                try {
                    unlinkSync(PID_FILE);
                } catch (unlinkErr: any) {
                    logger.error(`Failed to remove stale PID file (${PID_FILE}): ${unlinkErr.message}`);
                    // Still continue startup, but log the error
                }
                // Continue startup
            } else if (err.code === 'EPERM') {
                // Permission error - we can't check, safer to assume it might be running
                 logger.error(`🚫 Permission denied trying to check PID ${pid}. Cannot determine if server is running. Exiting to be safe.`);
                 process.exit(1); // Exit with error because state is uncertain
            } else {
                // Other unexpected error checking the process
                logger.error(`❌ Unexpected error checking PID ${pid}: ${err.message}. Exiting.`);
                process.exit(1); // Exit with error
            }
        }
    } else {
         logger.debug("No existing PID file found. Proceeding with startup.");
    }
}


// --- Cloud connection constants ---
const CLOUD_NAMESPACE = "/service";
const CLOUD_RECONNECT_DELAY_BASE_MS = 5000; // 5 seconds
// Allow null for infinite retries, mirroring Python's (None for infinite)
const CLOUD_MAX_RECONNECT_ATTEMPTS: number | null = null;
const CLOUD_MAX_RECONNECT_BACKOFF_MS = 60000; // 60 seconds

/**
 * Connects to the cloud server using Socket.IO
 * Uses command line arguments for configuration
 */
export const connectToCloud = (customLogger: winston.Logger, argv: any) => {
    // Use the passed logger for this function instead of global
    const logger = customLogger;
    if (argv['disable-cloud']) {
        logger.info("☁️ Cloud connection explicitly disabled via --disable-cloud.");
        return;
    }

    const cloudUrl = argv['cloud-url'];
    const email = argv['email'];
    const apiKey = argv['api-key'];
    const serviceName = argv['service-name'];

    if (!email || !apiKey || !serviceName) {
        logger.error("☁️❌ Missing required cloud connection arguments: --email, --api-key, --service-name. Cloud connection disabled.");
        return;
    }

    logger.info(`☁️ Attempting to connect to cloud server at ${cloudUrl}${CLOUD_NAMESPACE}...`);

    // Ensure previous socket is properly closed before creating a new one
    const existingSocket = getCloudSocket();
    if (existingSocket) {
        existingSocket.removeAllListeners(); // Clean up listeners
        existingSocket.disconnect();
        setCloudSocket(null);
    }
    const existingTimer = getCloudReconnectTimer();
    if (existingTimer) {
        clearTimeout(existingTimer); // Clear existing timer if any
        setCloudReconnectTimer(null);
    }

    // Construct the auth object matching server expectation
    const authPayload = { apiKey, email, serviceName, hostname: os.hostname(), port: argv.port }; // Add port here
    logger.debug(`☁️ Auth payload being sent: ${JSON.stringify(authPayload)}`); // Stringify explicitly

    // Create the socket with proper configuration
    const newSocket = io(`${cloudUrl}${CLOUD_NAMESPACE}`, {
        path: '/socket.io', // Standard path, adjust if cloud server uses something else
        transports: ['websocket'], // Prefer websocket
        autoConnect: false, // We'll connect manually after setting up handlers
        reconnection: true, // Enable built-in reconnection
        reconnectionAttempts: Infinity, // Retry forever
        reconnectionDelay: CLOUD_RECONNECT_DELAY_BASE_MS, // Initial delay
        reconnectionDelayMax: CLOUD_MAX_RECONNECT_BACKOFF_MS, // Max delay
        auth: authPayload,
    });
    
    // Save the socket in global state
    setCloudSocket(newSocket);

    newSocket.on('connect', () => {
        logger.info(`☁️✅ Successfully connected to cloud server: ${newSocket.id}`);
        resetCloudConnectionAttempts(); // Reset attempts on successful connect
        // Send identification to the cloud server
        if (serviceName) {
            const payload = { name: serviceName };
            newSocket.emit('client', payload);
            logger.debug('☁️⬆️ Sent client identification to cloud:', payload);
        } else {
            logger.warn('☁️⚠️ Cannot send client identification: service name is missing.');
        }
    });
    
    // Listen for tool_list direct events (compatibility with some cloud implementations)
    newSocket.on('tool_list', () => {
        logger.info('☁️✉️ Received tool_list direct event from cloud');
        // Import here to avoid circular dependency issues
        const { prepareToolsListPayload } = require('./dynamic');
        const tools = prepareToolsListPayload();
        logger.debug(`☁️ Sending ${tools.length} tools to cloud`);
        newSocket.emit('tools_updated', { tools });
    });
    
    // Listen for service_message JSON-RPC formatted requests
    newSocket.on('service_message', (data: any) => {
        logger.info('☁️✉️ Received service_message from cloud');  
        logger.debug(`☁️ Service message data: ${JSON.stringify(data)}`);
        
        // Check if this is a JSON-RPC request for tools/list
        if (data && data.jsonrpc === '2.0' && data.method === 'tools/list') {
            logger.info('☁️ Processing JSON-RPC tools/list request');
            const { prepareToolsListPayload } = require('./dynamic');
            const tools = prepareToolsListPayload();
            logger.debug(`☁️ Sending ${tools.length} tools via tools_updated`); 
            
            // Send tools_updated event with the tools
            newSocket.emit('tools_updated', { tools });
            
            // Also send a proper JSON-RPC response
            const response = {
                jsonrpc: '2.0',
                id: data.id,
                result: { tools }
            };
            newSocket.emit('mcp_response', response);
        }
    });

    newSocket.on('connect_error', (err: Error) => {
        logger.error(`☁️❌ Cloud connection error: ${err.message}`);
        // Check for specific auth errors if the server provides them
        if (err.message.includes('Authentication error')) { // Adjust based on actual server error
            logger.error("☁️ Authentication failed. Please check --email, --api-key, --service-name. Disabling cloud connection.");
            // Disable future attempts by disconnecting the socket if it exists
            newSocket.disconnect();
        } // Built-in reconnection will handle non-auth errors
    });

    newSocket.on('disconnect', (reason: string) => { // Disconnect reason is a string
        logger.warn(`☁️🔌 Disconnected from cloud server. Reason: ${reason}`);
        // Don't nullify socket reference here if we want auto-reconnect to work
        // Built-in reconnection will attempt unless reason is 'io client disconnect'
        if (reason === 'io server disconnect') {
            logger.info("☁️ Server initiated disconnect. Attempting to reconnect...");
            // Socket.IO will attempt reconnection automatically
        } else if (reason === 'io client disconnect') {
            logger.info(`☁️ Intentional disconnect from cloud. Won't reconnect automatically.`);
            // Socket is already disconnected, no need to call disconnect again.
        } else {
            logger.info(`☁️ Unexpected disconnect (${reason}). Socket.IO will attempt reconnection.`);
            // Socket.IO will attempt reconnection automatically for other reasons (e.g., transport error)
        }
    });

    // Handlers for reconnection events
    newSocket.on('reconnect_attempt', (attemptNumber: number) => {
        logger.info(`☁️ Retrying cloud connection... Attempt ${attemptNumber}`);
    });

    newSocket.on('reconnect', (attemptNumber: number) => {
        logger.info(`☁️✅ Successfully reconnected to cloud server after ${attemptNumber} attempts.`);
        // Client identification might need to be resent depending on server logic
        if (serviceName) {
            const payload = { name: serviceName };
            newSocket.emit('client', payload);
            logger.debug('☁️⬆️ Re-sent client identification after reconnect:', payload);
        }
    });

    newSocket.on('reconnect_error', (err: Error) => {
        logger.error(`☁️❌ Cloud reconnection error: ${err.message}`);
        // The library continues attempting based on reconnectionAttempts setting
    });

    newSocket.on('reconnect_failed', () => {
        // This event fires if reconnectionAttempts has a limit and it's reached
        // Since ours is Infinity, this should theoretically never fire
        logger.error("☁️❌ Cloud reconnection failed after maximum attempts (This shouldn't happen with infinite retries!).");
    });

    newSocket.on('error', (error: Error) => {
        logger.error(`☁️ Socket.IO general error: ${error.message}`);
        // General errors might not trigger disconnect
        if (!newSocket.connected) {
            // Built-in reconnection will handle this
        }
    });

    // Manually initiate the connection
    newSocket.connect();
    logger.debug('☁️ Connection initiated');
};

/**
 * Emit tools updated event to the cloud socket
 * @param tools The tools payload to send
 */
export const emitToolsUpdated = (tools: any[]): void => {
    // Uses cloudSocket from shared state
    const socket = getCloudSocket();
    if (socket && socket.connected) {
        socket.emit('tools_updated', { tools });
    }
};

/**
 * Disconnect from the cloud
 * @param logger The logger to use
 */
export const disconnectFromCloud = (customLogger: winston.Logger): void => {
    // Use the passed logger instead of global
    const logger = customLogger;
    const timer = getCloudReconnectTimer();
    if (timer) {
        clearTimeout(timer);
        setCloudReconnectTimer(null);
        logger.info("☁️ Cleared pending cloud reconnect timer.");
    }
    
    const socket = getCloudSocket();
    if (socket && socket.connected) {
        logger.info("☁️ Disconnecting from cloud server...");
        socket.disconnect(); // Trigger intentional disconnect
        setCloudSocket(null);
    }
};
