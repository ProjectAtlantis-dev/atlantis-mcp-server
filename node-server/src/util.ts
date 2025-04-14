import { existsSync, readFileSync, unlinkSync } from 'fs';
import process from 'process'; // Explicit import for process
import winston from 'winston'; // Import winston type for the logger parameter
import io from 'socket.io-client'; // Import for cloud connection
import os from 'os'; // Import for hostname access

// --- Function to Check Existing Process --- //
export function checkAndHandleExistingProcess(logger: winston.Logger, PID_FILE: string): void {
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


// --- Global state variables for cloud connection ---
let cloudSocket: any;
let cloudReconnectTimer: NodeJS.Timeout | null = null;
let cloudConnectionAttempts: number = 0;

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
export const connectToCloud = (logger: winston.Logger, argv: any) => {
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
    if (cloudSocket) {
        cloudSocket.removeAllListeners(); // Clean up listeners
        cloudSocket.disconnect();
        cloudSocket = null;
    }
    if (cloudReconnectTimer) {
        clearTimeout(cloudReconnectTimer); // Clear existing timer if any
        cloudReconnectTimer = null;
    }

    // Construct the auth object matching server expectation
    const authPayload = { apiKey, email, serviceName, hostname: os.hostname(), port: argv.port }; // Add port here
    logger.debug(`☁️ Auth payload being sent: ${JSON.stringify(authPayload)}`); // Stringify explicitly

    cloudSocket = io(`${cloudUrl}${CLOUD_NAMESPACE}`, { // Connect directly to the namespace URL
        path: '/socket.io', // Standard path, adjust if cloud server uses something else
        transports: ['websocket'], // Prefer websocket
        autoConnect: true, // We want autoConnect for reconnection
        reconnection: true, // Enable built-in reconnection
        reconnectionAttempts: Infinity, // Retry forever
        reconnectionDelay: CLOUD_RECONNECT_DELAY_BASE_MS, // Initial delay
        reconnectionDelayMax: CLOUD_MAX_RECONNECT_BACKOFF_MS, // Max delay
        auth: authPayload,
        // Specify the namespace if needed, often done in the URL or path
        // For Socket.IO v3/v4, namespace is usually part of the URL or handled server-side
        // If connection fails, try adding namespace to URL: io(`${cloudUrl}${CLOUD_NAMESPACE}`, {...})
    });

    cloudSocket.on('connect', () => {
        logger.info(`☁️✅ Successfully connected to cloud server: ${cloudSocket?.id}`);
        cloudConnectionAttempts = 0; // Reset attempts on successful connect
        // Send identification to the cloud server
        const serviceName = argv['service-name'];
        if (serviceName && cloudSocket) { // Ensure socket exists and we have a name
            const payload = { name: serviceName };
            cloudSocket.emit('client', payload);
            logger.debug('☁️⬆️ Sent client identification to cloud:', payload);
        } else if (!serviceName) {
            logger.warn('☁️⚠️ Cannot send client identification: service name is missing.');
        }
        // TODO: Send any necessary identification or registration messages?
    });

    cloudSocket.on('connect_error', (err: Error) => {
        logger.error(`☁️❌ Cloud connection error: ${err.message}`);
        // Check for specific auth errors if the server provides them
        if (err.message.includes('Authentication error')) { // Adjust based on actual server error
            logger.error("☁️ Authentication failed. Please check --email, --api-key, --service-name. Disabling cloud connection.");
            // Disable future attempts by disconnecting the socket if it exists
            cloudSocket?.disconnect();
        } // Built-in reconnection will handle non-auth errors
    });

    cloudSocket.on('disconnect', (reason: string) => { // Disconnect reason is a string
        logger.warn(`☁️🔌 Disconnected from cloud server. Reason: ${reason}`);
        // Don't nullify cloudSocket here if we want auto-reconnect to work
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

    // --- ADDED: Handlers for built-in reconnection events ---
    cloudSocket.on('reconnect_attempt', (attemptNumber: number) => {
        logger.info(`☁️ Retrying cloud connection... Attempt ${attemptNumber}`);
    });

    cloudSocket.on('reconnect', (attemptNumber: number) => {
        logger.info(`☁️✅ Successfully reconnected to cloud server after ${attemptNumber} attempts.`);
        // No need to reset attempts manually, library handles it.
        // Client identification might need to be resent depending on server logic.
        // Re-emitting identification on 'reconnect' is often a good idea.
        const serviceName = argv['service-name'];
        if (serviceName && cloudSocket) {
            const payload = { name: serviceName };
            cloudSocket.emit('client', payload);
            logger.debug('☁️⬆️ Re-sent client identification after reconnect:', payload);
        }
    });

    cloudSocket.on('reconnect_error', (err: Error) => {
        logger.error(`☁️❌ Cloud reconnection error: ${err.message}`);
        // The library continues attempting based on reconnectionAttempts setting.
    });

    cloudSocket.on('reconnect_failed', () => {
        // This event fires if reconnectionAttempts has a limit and it's reached.
        // Since ours is Infinity, this should theoretically never fire.
        logger.error("☁️❌ Cloud reconnection failed after maximum attempts (This shouldn't happen with infinite retries!).");
    });
    // --- End Added Handlers ---

    cloudSocket.on('error', (error: Error) => {
        logger.error(`☁️ Socket.IO general error: ${error.message}`);
        // General errors might not trigger disconnect, schedule check/reconnect
        if (!cloudSocket?.connected) {
             // Built-in reconnection will handle this
        }
    });

    // Manually initiate the connection
    cloudSocket.connect();
};


