// node-server/server.js
const WebSocket = require('ws');
const path = require('path');
const fs = require('fs');
const winston = require('winston');
const { v4: uuidv4 } = require('uuid'); // For generating message IDs later

// --- Configuration (Defaults, will be overridden by args later) ---
const HOST = '0.0.0.0';
const PORT = 8001; // Default port, different from Python server
const MCP_PATH = '/mcp'; // Standard MCP path
const FUNCTIONS_DIR = path.join(__dirname, 'dynamic_functions');
const PID_FILE = path.join(__dirname, 'mcp_server.pid');

// --- Logging Setup ---
const logger = winston.createLogger({
    level: 'debug', // Log debug level and above
    format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.printf(info => `${info.timestamp} [${info.level.toUpperCase()}] mcp-node-server: ${info.message}`)
    ),
    transports: [
        new winston.transports.Console()
        // TODO: Add file transport later if needed
    ]
});

logger.info("🔧 Initializing MCP Node Server...");

// --- Helper Functions ---
const sendResponse = (ws, id, result) => {
    const response = {
        jsonrpc: "2.0",
        id: id,
        result: result
    };
    const responseString = JSON.stringify(response);
    ws.send(responseString);
    const clientId = clients.get(ws) || 'unknown';
    logger.debug(`⬆️ Sent response to ${clientId} (ID: ${id}): ${responseString}`);
};

const sendError = (ws, id, code, message, data = null) => {
    const errorResponse = {
        jsonrpc: "2.0",
        id: id,
        error: {
            code: code,
            message: message,
            ...(data && { data: data }) // Include data if provided
        }
    };
    const errorString = JSON.stringify(errorResponse);
    ws.send(errorString);
    const clientId = clients.get(ws) || 'unknown';
    logger.debug(`⬆️ Sent error to ${clientId} (ID: ${id}): ${errorString}`);
};

// --- MCP Tool Registry (will be populated dynamically) ---
const toolRegistry = new Map();

// --- MCP Handlers ---
const handleListTools = (ws, id, params) => {
    // TODO: Populate this with actual built-in and dynamic tools
    const tools = Array.from(toolRegistry.values());
    logger.info(`🛠️ Responding to tools/list request (ID: ${id}) with ${tools.length} tools.`);
    sendResponse(ws, id, { tools: tools });
};

const handleCallTool = (ws, id, params) => {
    const { name, args } = params || {};

    if (!name) {
        return sendError(ws, id, -32602, "Invalid params: 'name' is required for tools/call");
    }

    logger.info(`📞 Received tools/call request for tool: ${name} (ID: ${id})`);

    // TODO: Implement actual tool execution
    const toolFunction = toolRegistry.get(name);

    if (toolFunction) {
        // Placeholder for actual execution
        logger.warn(`⚠️ Tool '${name}' found but execution not yet implemented.`);
        sendError(ws, id, -32601, `Tool execution not implemented: ${name}`);
        // try {
        //     const result = await toolFunction(args); // Assuming async tool functions
        //     sendResponse(ws, id, { content: result }); // Adjust response format as needed by MCP spec
        // } catch (error) {
        //     logger.error(`💥 Error executing tool '${name}': ${error.message}`);
        //     sendError(ws, id, -32000, `Server error during tool execution: ${error.message}`);
        // }
    } else {
        logger.warn(`❓ Tool '${name}' not found in registry.`);
        sendError(ws, id, -32601, `Method not found: Tool '${name}' is not registered.`);
    }
};

// Add more handlers as needed (prompts/list, resources/list, notifications?)

// --- Ensure Dynamic Functions Directory Exists ---
if (!fs.existsSync(FUNCTIONS_DIR)) {
    logger.info(`📁 Creating dynamic functions directory: ${FUNCTIONS_DIR}`);
    fs.mkdirSync(FUNCTIONS_DIR, { recursive: true });
}

// --- WebSocket Server Setup ---
// We need to handle the specific '/mcp' path
const server = require('http').createServer(); // Create HTTP server to attach WebSocket server
const wss = new WebSocket.Server({ noServer: true }); // Don't start WebSocket server automatically

const clients = new Map(); // Store connected clients (ws -> clientId)

wss.on('connection', (ws, request, clientId) => {
    logger.info(`✅ Client connected: ${clientId} from ${request.socket.remoteAddress}`);
    clients.set(ws, clientId);

    ws.on('message', (message) => {
        let parsedMessage;
        let messageId = null; // Keep track of ID for potential error messages
        try {
            const messageString = message.toString();
            logger.debug(`⬇️ Received message from ${clientId}: ${messageString}`);
            parsedMessage = JSON.parse(messageString);
            messageId = parsedMessage.id !== undefined ? parsedMessage.id : null; // Handle notifications without IDs

            // Basic JSON-RPC validation
            if (parsedMessage.jsonrpc !== "2.0" || !parsedMessage.method) {
                // Attempt to send error only if it's not a notification
                if(messageId !== null) {
                    sendError(ws, messageId, -32600, "Invalid Request: Missing 'jsonrpc' or 'method'.");
                } else {
                    logger.warn(`Received invalid non-request message (no ID): ${messageString}`);
                }
                return;
            }

            // Route based on method
            switch (parsedMessage.method) {
                case "tools/list":
                    handleListTools(ws, messageId, parsedMessage.params);
                    break;
                case "tools/call":
                    handleCallTool(ws, messageId, parsedMessage.params);
                    break;
                // TODO: Add cases for prompts/list, resources/list, etc.
                default:
                    logger.warn(`❓ Received unknown method: ${parsedMessage.method}`);
                    if (messageId !== null) { // Don't send error for notifications
                        sendError(ws, messageId, -32601, `Method not found: ${parsedMessage.method}`);
                    }
            }

        } catch (error) {
            logger.error(`❌ Error processing message from ${clientId}: ${error.message}`);
            logger.debug(error.stack);
            // Attempt to send error only if it's not a notification and ID was parsed
            if(error instanceof SyntaxError){
                // Send Parse error (-32700) if JSON was invalid
                // We don't know the ID if parsing failed, so pass null
                sendError(ws, null, -32700, "Parse error: Invalid JSON received.");
            } else if (messageId !== null) {
                // Send Internal error (-32603) for other processing errors
                sendError(ws, messageId, -32603, `Internal error: ${error.message}`);
            }
        }
    });

    ws.on('close', (code, reason) => {
        const reasonString = reason ? reason.toString() : 'No reason given';
        logger.info(`❌ Client disconnected: ${clientId}. Code: ${code}, Reason: ${reasonString}`);
        clients.delete(ws);
    });

    ws.on('error', (error) => {
        logger.error(`💥 WebSocket error for client ${clientId}: ${error.message}`);
        clients.delete(ws); // Ensure cleanup on error
    });

    // Send initial connection confirmation (optional)
    // logger.info(`✉️ Sent connection confirmation to ${clientId}`);
    // ws.send(JSON.stringify({ type: 'connection_ack', clientId: clientId }));
});

// --- HTTP Server Upgrade Handling ---
// Handle requests and upgrade only those for the correct path
server.on('upgrade', (request, socket, head) => {
    const { pathname } = new URL(request.url, `http://${request.headers.host}`);

    if (pathname === MCP_PATH) {
        const clientId = uuidv4(); // Generate a unique ID for this client connection
        logger.debug(`🔌 Handling upgrade request for ${MCP_PATH} from ${socket.remoteAddress}, assigning ID: ${clientId}`);
        wss.handleUpgrade(request, socket, head, (ws) => {
            wss.emit('connection', ws, request, clientId);
        });
    } else {
        logger.warn(`🚫 Rejecting upgrade request for invalid path: ${pathname}`);
        socket.destroy();
    }
});

// --- Start Listening ---
server.listen(PORT, HOST, () => {
    logger.info(`🌟 MCP NODE SERVER LISTENING ON ws://${HOST}:${PORT}${MCP_PATH}`);
    // TODO: Add PID file creation here
});

// --- Graceful Shutdown Handling (Initial Setup) ---
// TODO: Implement proper PID file check/removal and signal handling
process.on('SIGINT', () => {
    logger.info("🚦 Received SIGINT. Shutting down gracefully...");
    // TODO: Close WebSocket connections
    // TODO: Stop cloud connection if active
    // TODO: Remove PID file
    server.close(() => {
        logger.info("🛑 Server closed.");
        process.exit(0);
    });
    // Force exit after a timeout if graceful shutdown fails
    setTimeout(() => {
        logger.warn("⚠️ Graceful shutdown timed out. Forcing exit.");
        process.exit(1);
    }, 5000); // 5 seconds timeout
});

process.on('SIGTERM', () => {
    logger.info("🚦 Received SIGTERM. Shutting down gracefully...");
    // Same shutdown logic as SIGINT
    process.emit('SIGINT');
});

// --- Basic Error Handling ---
process.on('uncaughtException', (error) => {
    logger.error(`💥 UNCAUGHT EXCEPTION: ${error.message}`);
    logger.error(error.stack);
    // TODO: Attempt cleanup (PID?) before exiting
    process.exit(1); // Exit forcefully on uncaught exceptions
});

process.on('unhandledRejection', (reason, promise) => {
    logger.error('💥 UNHANDLED REJECTION at:', promise, 'reason:', reason);
    // TODO: Attempt cleanup (PID?) before exiting
    process.exit(1); // Exit forcefully on unhandled rejections
});
