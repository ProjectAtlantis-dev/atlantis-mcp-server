import WebSocket, { WebSocketServer } from 'ws';
import http from 'http';
import url from 'url';
import winston from 'winston';
import path from 'path';
import fs from 'fs/promises';
import { existsSync, readFileSync, writeFileSync, unlinkSync } from 'fs';

// --- Import Shared Types ---
import {
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    TextContent,
    ToolDefinition,
    ToolParameterProperty, // Needed for stub creation
    DynamicFunctionModule
} from './types';

// --- Logger Setup ---
const logger = winston.createLogger({
    level: 'info', // TODO: Make configurable via args
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.printf(({ timestamp, level, message }) => `${timestamp} ${level.toUpperCase()}: ${message}`)
    ),
    transports: [ new winston.transports.Console() ],
});

// --- Constants ---
const HOST: string = process.env.HOST || '0.0.0.0'; // TODO: Use yargs
const PORT: number = parseInt(process.env.PORT || '8001', 10); // TODO: Use yargs
const MCP_PATH: string = '/mcp';
// Points to the DIRECTORY where compiled JS function files will reside
const COMPILED_FUNCTIONS_DIR: string = path.join(__dirname, 'dynamic_functions');
// Base source directory for creating the compiled dir if needed
const SOURCE_FUNCTIONS_DIR: string = path.join(__dirname, '..', 'src', 'dynamic_functions');
const PID_FILE: string = path.join(__dirname, '..', 'mcp_node_server.pid'); // Place in project root (outside dist)

// --- MCP Tool Registry ---
const toolRegistry = new Map<string, ToolDefinition>();
// Keep track of file path per dynamic function name (using compiled path)
const dynamicFunctionFiles = new Map<string, string>();

// --- Helper Functions ---

const sendResponse = (ws: WebSocket, id: string | number | null, result: any): void => {
    const response: JsonRpcResponse = { jsonrpc: '2.0', result, id };
    try { ws.send(JSON.stringify(response)); } catch (e: any) { logger.error(`Failed to send response: ${e.message}`); }
};

const sendError = (ws: WebSocket, id: string | number | null, code: number, message: string, data?: any): void => {
    const error: JsonRpcError = { code, message, data };
    const response: JsonRpcResponse = { jsonrpc: '2.0', error, id };
     try { ws.send(JSON.stringify(response)); } catch (e: any) { logger.error(`Failed to send error response: ${e.message}`); }
};

// --- Dynamic Function Loading ---
const loadAndRegisterDynamicFunction = async (compiledFilePath: string): Promise<void> => {
    // Use compiled JS file path
    const functionName = path.basename(compiledFilePath, '.js');

    if (toolRegistry.has(functionName) && !dynamicFunctionFiles.has(functionName)) {
        logger.warn(`⚠️ Skipping dynamic load for '${functionName}': name conflicts with a built-in/stub tool.`);
        return;
    }

    logger.debug(`Attempting to load dynamic function '${functionName}' from ${compiledFilePath}`);

    try {
        // Ensure cache is clear for the specific file being loaded/reloaded
        const absolutePath = require.resolve(compiledFilePath);
        delete require.cache[absolutePath];
        logger.debug(`Cleared require cache for: ${absolutePath}`)

        const dynamicModule: DynamicFunctionModule = require(absolutePath);

        // Validate Module Structure (using types)
        if (typeof dynamicModule.handler !== 'function') throw new Error(`Module does not export a 'handler' function.`);
        if (typeof dynamicModule.metadata !== 'object' || dynamicModule.metadata === null) throw new Error(`Module does not export a 'metadata' object.`);
        if (typeof dynamicModule.metadata.description !== 'string' || !dynamicModule.metadata.description) throw new Error(`Exported 'metadata' missing required 'description' string.`);

        // Default parameters if missing
        const parameters = dynamicModule.metadata.parameters ?? { type: 'object', properties: {} };
         if (!dynamicModule.metadata.parameters) {
            logger.debug(`Function '${functionName}' has no parameters defined in metadata.`);
         }


        const toolDef: ToolDefinition = {
            name: functionName,
            description: dynamicModule.metadata.description,
            parameters: parameters,
            _function: dynamicModule.handler, // Store the actual function
            _filePath: compiledFilePath // Store the path to the loaded JS file
        };

        toolRegistry.set(functionName, toolDef);
        dynamicFunctionFiles.set(functionName, compiledFilePath);

        logger.info(`✅ Dynamically registered function '${functionName}'`);

    } catch (error: any) {
        logger.error(`❌ Failed to load dynamic function '${functionName}' from ${compiledFilePath}: ${error.message}`);
        if (error.stack) logger.debug(error.stack);
        // If it was previously registered, remove it
        if (dynamicFunctionFiles.has(functionName)) {
            toolRegistry.delete(functionName);
            dynamicFunctionFiles.delete(functionName);
        }
    }
};


// --- Stub Tool Definitions ---
const createStubTool = (
    name: string,
    description: string,
    parametersSchema: { [key: string]: ToolParameterProperty },
    requiredParams: string[] = []
): ToolDefinition => {
    return {
        name: name,
        description: description + " (STUB)",
        parameters: { type: "object", properties: parametersSchema, required: requiredParams },
        async execute(args: any): Promise<TextContent[]> {
            logger.info(`Executing STUB for tool: ${name} with args: ${JSON.stringify(args)}`);
            return [{ type: "text", text: `Tool '${name}' called successfully (stub implementation).` }];
        }
    };
};

// Define stubs... (keep these as before)
const registerFunctionStub = createStubTool("_register_function", "Registers a new dynamic function.", { function_name: { type: "string" }, code: { type: "string" } }, ["function_name", "code"]);
const getFunctionCodeStub = createStubTool("_get_function_code", "Gets source code of a dynamic function.", { function_name: { type: "string" } }, ["function_name"]);
const removeFunctionStub = createStubTool("_remove_function", "Removes a dynamic function.", { function_name: { type: "string" } }, ["function_name"]);
const taskAddStub = createStubTool("_task_add", "Adds a new task.", { task_id: { type: "string" }, payload: { type: "object" }, schedule: { type: "string"} }, ["task_id", "payload"]);
const taskRunStub = createStubTool("_task_run", "Runs a task.", { task_id: { type: "string" } }, ["task_id"]);
const taskRemoveStub = createStubTool("_task_remove", "Removes a task.", { task_id: { type: "string" } }, ["task_id"]);
const taskPeekStub = createStubTool("_task_peek", "Gets task details.", { task_id: { type: "string" } }, ["task_id"]);

// Register stubs...
toolRegistry.set(registerFunctionStub.name, registerFunctionStub);
toolRegistry.set(getFunctionCodeStub.name, getFunctionCodeStub);
toolRegistry.set(removeFunctionStub.name, removeFunctionStub);
toolRegistry.set(taskAddStub.name, taskAddStub);
toolRegistry.set(taskRunStub.name, taskRunStub);
toolRegistry.set(taskRemoveStub.name, taskRemoveStub);
toolRegistry.set(taskPeekStub.name, taskPeekStub);
// logger.info(`🛠️ Registered ${toolRegistry.size} built-in stub tools initially.`); // Logged later after scan


// --- MCP Request Handlers ---

const handleListTools = (ws: WebSocket, id: string | number | null, _params: any): void => {
    const toolsForClient = Array.from(toolRegistry.values()).map(tool => {
        const { _function, _filePath, execute, ...toolDefinition } = tool;
        return toolDefinition;
    });
    logger.info(`🛠️ Responding to tools/list request (ID: ${id}) with ${toolsForClient.length} tools.`);
    sendResponse(ws, id, { tools: toolsForClient });
};

const handleCallTool = async (ws: WebSocket, id: string | number | null, params: any): Promise<void> => {
    const { name, args } = params || {};
    if (!name) return sendError(ws, id, -32602, "Invalid params: 'name' is required");

    logger.info(`📞 Received tools/call: ${name} (ID: ${id}) args: ${JSON.stringify(args)}`);
    const tool = toolRegistry.get(name);

    if (tool) {
        try {
            // TODO: Add proper jsonschema validation here
            if (tool.parameters && Array.isArray(tool.parameters.required)) {
                for (const requiredParam of tool.parameters.required) {
                    if (!args || args[requiredParam] === undefined) {
                        throw new Error(`Missing required parameter: '${requiredParam}'`);
                    }
                }
            }

            let result: TextContent[];

            if (typeof tool._function === 'function') {
                logger.debug(`Executing dynamic function '${name}'`);
                const dynamicResult = await tool._function(args || {});
                if (Array.isArray(dynamicResult) && dynamicResult.length > 0 && dynamicResult[0].type === 'text') {
                    result = dynamicResult as TextContent[]; // Assume correct format
                } else if (typeof dynamicResult === 'string' || typeof dynamicResult === 'number') {
                    result = [{ type: 'text', text: String(dynamicResult) }];
                } else {
                     logger.error(`💥 Dynamic function '${name}' returned unexpected format: ${JSON.stringify(dynamicResult)}`);
                     throw new Error(`Tool '${name}' function returned unexpected format.`);
                }
            } else if (typeof tool.execute === 'function') {
                logger.debug(`Executing built-in/stub tool '${name}'`);
                result = await tool.execute(args || {});
            } else {
                throw new Error(`Tool '${name}' has no execution method.`);
            }

            if (!Array.isArray(result) || !result.every(item => item?.type === 'text' && typeof item.text === 'string')) {
                 logger.error(`💥 Tool '${name}' execution returned invalid format. Expected Array<TextContent>. Got: ${JSON.stringify(result)}`);
                 return sendError(ws, id, -32603, `Internal error: Tool '${name}' returned invalid result format.`);
            }

            logger.info(`✅ Successfully executed tool '${name}' (ID: ${id}).`);
            sendResponse(ws, id, { content: result });

        } catch (error: any) {
            logger.error(`💥 Error executing tool '${name}': ${error.message}`);
            if (error.stack) logger.debug(error.stack);
            const errorCode = error.message?.startsWith("Missing required parameter") ? -32602 : -32000;
            sendError(ws, id, errorCode, `Server error: ${error.message}`);
        }
    } else {
        logger.warn(`❓ Tool '${name}' not found.`);
        sendError(ws, id, -32601, `Method not found: Tool '${name}'`);
    }
};

// --- Main WebSocket Message Handling ---
const handleMessage = (ws: WebSocket, message: Buffer): void => {
    let request: JsonRpcRequest;
    try {
        request = JSON.parse(message.toString());
        logger.debug(`Received raw: ${message.toString()}`);
        if (request.jsonrpc !== '2.0' || !request.method || typeof request.id === 'undefined') throw new Error("Invalid JSON-RPC.");
    } catch (e: any) {
        logger.error(`❌ Invalid message: ${message.toString()}`);
        let id = null; try { id = JSON.parse(message.toString()).id; } catch { /* ignore */ }
        return sendError(ws, id ?? null, -32700, "Parse error", e.message);
    }

    switch (request.method) {
        case 'tools/list':
            handleListTools(ws, request.id, request.params);
            break;
        case 'tools/call':
            handleCallTool(ws, request.id, request.params).catch(err => {
                logger.error(`🚨 Unhandled error in handleCallTool for ${request.method} (ID: ${request.id}): ${err.message}`, { stack: err.stack });
                sendError(ws, request.id, -32000, "Internal server error.");
            });
            break;
        default:
            logger.warn(`❓ Unknown method: ${request.method}`);
            sendError(ws, request.id, -32601, "Method not found");
    }
};

// --- HTTP & WebSocket Server Setup ---
const server = http.createServer((_req, res) => { res.writeHead(404).end('Not Found'); });
const wss = new WebSocketServer({ noServer: true });

wss.on('connection', (ws: WebSocket, request: http.IncomingMessage) => {
    const clientAddr = request.socket.remoteAddress || 'unknown';
    logger.info(`➕ Client connected: ${clientAddr}`);
    ws.on('message', (message: Buffer) => handleMessage(ws, message));
    ws.on('close', () => logger.info(`➖ Client disconnected: ${clientAddr}`));
    ws.on('error', (error: Error) => logger.error(`❗️ WS error (${clientAddr}): ${error.message}`));
});

server.on('upgrade', (request: http.IncomingMessage, socket: any, head: Buffer) => {
    const pathname = request.url ? url.parse(request.url).pathname : undefined;
    if (pathname === MCP_PATH) {
        wss.handleUpgrade(request, socket, head, (ws) => wss.emit('connection', ws, request));
    } else {
        logger.warn(`🚫 Rejecting upgrade for path: ${pathname}`);
        socket.destroy();
    }
});

// --- Start Server & Initial Setup ---
const startServer = async () => {
    logger.info(`📝 Checking PID file: ${PID_FILE}`);
    if (existsSync(PID_FILE)) {
        try {
            const pidString = readFileSync(PID_FILE, 'utf8');
            const pid = parseInt(pidString, 10);
            if (!isNaN(pid)) {
                 // Check if process is running. Signal 0 just checks existence.
                 try {
                     process.kill(pid, 0);
                     // If the above doesn't throw, the process exists
                     logger.error(`❌ Server already running with PID ${pid} (found in ${PID_FILE}). Exiting.`);
                     process.exit(1);
                 } catch (e: any) {
                     if (e.code === 'ESRCH') {
                         // Process doesn't exist, stale PID file
                         logger.warn(`⚠️ Found stale PID file (${PID_FILE}) for non-existent process ${pid}. Removing it.`);
                         try { unlinkSync(PID_FILE); } catch (unlinkErr: any) { logger.error(`Failed to remove stale PID file: ${unlinkErr.message}`); }
                     } else {
                         // Other error (e.g., permissions)
                         logger.error(`❌ Error checking PID ${pid} from ${PID_FILE}: ${e.message}. Exiting.`);
                         process.exit(1);
                     }
                 }
            } else {
                 logger.warn(`⚠️ PID file (${PID_FILE}) contains invalid content: "${pidString}". Ignoring and removing.`);
                 try { unlinkSync(PID_FILE); } catch (unlinkErr: any) { logger.error(`Failed to remove invalid PID file: ${unlinkErr.message}`); }
            }
        } catch (readErr: any) {
            logger.error(`❌ Error reading PID file (${PID_FILE}): ${readErr.message}. Exiting.`);
            process.exit(1);
        }
    }

    logger.info(`📂 Source dynamic functions expected in: ${SOURCE_FUNCTIONS_DIR}`);
    logger.info(`📂 Compiled dynamic functions expected in: ${COMPILED_FUNCTIONS_DIR}`);
    // Ensure the COMPILED directory exists before scanning
    try {
        await fs.mkdir(COMPILED_FUNCTIONS_DIR, { recursive: true });
        logger.info(`✅ Ensured compiled dynamic functions directory exists.`);

        // Scan the COMPILED directory
        logger.info(`🔍 Scanning for initial dynamic functions in ${COMPILED_FUNCTIONS_DIR}...`);
        const files = await fs.readdir(COMPILED_FUNCTIONS_DIR);
        const functionFiles = files.filter(file => file.endsWith('.js'));

        if (functionFiles.length > 0) {
            logger.info(`Found ${functionFiles.length} potential function files.`);
            for (const file of functionFiles) {
                const filePath = path.join(COMPILED_FUNCTIONS_DIR, file);
                await loadAndRegisterDynamicFunction(filePath);
            }
        } else {
            logger.info(`No initial dynamic functions found.`);
        }
    } catch (error: any) {
        logger.error(`❌ Failed during initial dynamic function setup: ${error.message}`);
        // Consider exiting if this fails critically
    }

     logger.info(`🛠️ Total tools registered after scan: ${toolRegistry.size}`);


    server.listen(PORT, HOST, () => {
        logger.info(`🌟 MCP NODE SERVER (TypeScript) LISTENING ON ws://${HOST}:${PORT}${MCP_PATH}`);
        // --- Write PID file on successful start ---
        try {
            writeFileSync(PID_FILE, String(process.pid), 'utf8');
            logger.info(`📝 Created PID file: ${PID_FILE} with PID ${process.pid}`);
        } catch (writeErr: any) {
            logger.error(`❌ Failed to write PID file (${PID_FILE}): ${writeErr.message}. Server is running but PID file is missing.`);
            // Continue running, but log the error
        }
    });
};

// --- Graceful Shutdown ---
const shutdown = async (): Promise<void> => {
    let exitCode = 0; // Default to success
    logger.info("🚦 Shutting down gracefully...");
    try {
        // Main shutdown operations
        server.close(() => logger.info('🛑 HTTP server closed.'));
        logger.info(`🔌 Closing ${wss.clients.size} client connections...`);
        wss.clients.forEach(client => { if (client.readyState === WebSocket.OPEN) client.close(); });

        // Add other cleanup (cloud connection, etc.)
        await new Promise(resolve => setTimeout(resolve, 500)); // Brief pause
        logger.info("✅ Main shutdown operations complete.");

    } catch (shutdownErr: any) {
        logger.error(`❌ Error during shutdown sequence: ${shutdownErr.message}`);
        exitCode = 1; // Indicate an error occurred
    } finally {
        // --- Remove PID file ---
        // This runs regardless of errors in the try block
        logger.info("🧹 Attempting PID file cleanup...");
        try {
            if (existsSync(PID_FILE)) {
                unlinkSync(PID_FILE);
                logger.info(`🗑️ Removed PID file: ${PID_FILE}`);
            } else {
                logger.info(`🤷 PID file not found, nothing to remove.`);
            }
        } catch (unlinkErr: any) {
            logger.error(`⚠️ Failed during PID file removal (${PID_FILE}) on shutdown: ${unlinkErr.message}`);
            // Don't change exit code here, the primary shutdown might have succeeded/failed already
        }

        logger.info(`👋 Server shutdown process finished with exit code ${exitCode}.`);
        process.exit(exitCode); // Exit with appropriate code
    }
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
process.on('uncaughtException', (error: Error) => { logger.error(`🚨 UNCAUGHT EXCEPTION: ${error.message}`, { stack: error.stack }); /* Consider shutdown(1); */ });
process.on('unhandledRejection', (reason: any) => { logger.error('🚨 UNHANDLED REJECTION:', { reason }); /* Consider shutdown(1); */ });

// --- Run the server ---
startServer().catch(error => {
    logger.error(`🚨 Failed to start server: ${error.message}`, { stack: error.stack });
    process.exit(1);
});
