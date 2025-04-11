import WebSocket, { WebSocketServer } from 'ws';
import http from 'http';
import url from 'url';
import winston from 'winston';
import path from 'path';
import fs from 'fs/promises';
import { existsSync, readFileSync, writeFileSync, unlinkSync } from 'fs';
import io from 'socket.io-client'; // Standard import for the function
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import os from 'os'; // ADDED IMPORT
import process from 'process'; // Explicit import for process.kill

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

// --- Argument Parsing ---
const argv = yargs(hideBin(process.argv)).options({
    'email': { type: 'string', description: 'Cloud server email for authentication' },
    'api-key': { type: 'string', description: 'Cloud server API key for authentication' },
    'service-name': { type: 'string', description: 'Service name to register with cloud server' },
    'cloud-url': { type: 'string', default: "http://localhost:3010", description: 'URL of the cloud server' },
    'disable-cloud': { type: 'boolean', default: false, description: 'Disable cloud connection' },
    'host': { type: 'string', default: '0.0.0.0', description: 'Host for the local server' },
    'port': { type: 'number', default: 8001, description: 'Port for the local server' },
    'log-level': { type: 'string', default: 'debug', choices: ['error', 'warn', 'info', 'verbose', 'debug', 'silly'], description: 'Logging level' } // Default to debug
}).alias('e', 'email').alias('k', 'api-key').alias('s', 'service-name').alias('c', 'cloud-url').help().parseSync();

// --- Logger Setup ---
const logger = winston.createLogger({
    level: argv['log-level'] || 'info', // TODO: Make configurable via args
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.printf(({ timestamp, level, message }) => `${timestamp} ${level.toUpperCase()}: ${message}`)
    ),
    transports: [ new winston.transports.Console() ],
});

// --- Constants ---
const HOST: string = argv['host']; // TODO: Use yargs
const PORT: number = argv['port']; // TODO: Use yargs
const MCP_PATH: string = '/mcp';
// Points to the DIRECTORY where compiled JS function files will reside
const COMPILED_FUNCTIONS_DIR: string = path.join(__dirname, '..', 'dynamic_functions'); // Point outside src
// Base source directory for creating the compiled dir if needed
const SOURCE_FUNCTIONS_DIR: string = path.join(__dirname, '..', 'dynamic_functions'); // Correct path pointing outside src
const PID_FILE: string = path.join(__dirname, '..', 'mcp_node_server.pid'); // Place in project root (outside dist)
const CLOUD_NAMESPACE = "/service";
const CLOUD_RECONNECT_DELAY_BASE_MS = 5000; // 5 seconds
// Allow null for infinite retries, mirroring Python's (None for infinite)
const CLOUD_MAX_RECONNECT_ATTEMPTS: number | null = 10;
const CLOUD_MAX_RECONNECT_BACKOFF_MS = 60000; // 60 seconds

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
        const inputSchema = dynamicModule.metadata.inputSchema ?? { type: 'object', properties: {} };

         if (!dynamicModule.metadata.inputSchema) {
            logger.debug(`Function '${functionName}' has no input schema defined in metadata.`);
         }


        const toolDef: ToolDefinition = {
            name: functionName,
            description: dynamicModule.metadata.description,
            inputSchema: inputSchema,
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
    inputSchema: { [key: string]: ToolParameterProperty },
    requiredParams: string[] = []
): ToolDefinition => {
    return {
        name: name,
        description: description + " (STUB)",
        inputSchema: { type: "object", properties: inputSchema, required: requiredParams },
        async execute(args: any): Promise<TextContent[]> {
            logger.info(`Executing STUB for tool: ${name} with args: ${JSON.stringify(args)}`);
            return [{ type: "text", text: `Tool '${name}' called successfully (stub implementation).` }];
        }
    };
};

// Define stubs... (keep these as before)
const registerFunctionStub = createStubTool("_register_function", "Registers a new dynamic function.", { function_name: { type: "string" }, code: { type: "string" } }, ["function_name", "code"]);

// --- Custom implementation for _function_get ---
const getFunctionCodeTool: ToolDefinition = {
    name: "_function_get",
    description: "Gets the TypeScript source code for a dynamic function.",
    inputSchema: {
        type: "object",
        properties: {
            function_name: { type: "string", description: "The name of the function (without .ts extension)" }
        },
        required: ["function_name"]
    },
    async execute(args: { function_name?: string }): Promise<TextContent[]> {
        const functionName = args.function_name;
        if (!functionName) {
            throw new Error("Missing required argument: function_name");
        }
        // Construct path assuming .ts files are in SOURCE_FUNCTIONS_DIR
        const sourceFilePath = path.join(SOURCE_FUNCTIONS_DIR, `${functionName}.ts`);
        logger.info(`Attempting to read source code from: ${sourceFilePath}`);

        try {
            const exists = existsSync(sourceFilePath); // Use sync existsSync for quick check
            if (!exists) {
                logger.warn(`Source file not found: ${sourceFilePath}`);
                throw new Error(`Function source file '${functionName}.ts' not found.`);
            }
            const fileContent = await fs.readFile(sourceFilePath, 'utf8');
            logger.info(`Successfully read source code for '${functionName}'.`);
            return [{ type: "text", text: fileContent }];
        } catch (error: any) {
            logger.error(`Error reading source file ${sourceFilePath}: ${error.message}`);
            // Re-throw for the main handler to catch and send appropriate JSON-RPC error
            throw new Error(`Failed to get source code for function '${functionName}': ${error.message}`);
        }
    }
};

const removeFunctionStub = createStubTool("_remove_function", "Removes a dynamic function.", { function_name: { type: "string" } }, ["function_name"]);
const taskAddStub = createStubTool("_task_add", "Adds a task.", { payload: { type: "object" } }, ["payload"]); // Simplified task stubs based on Python
const taskRunStub = createStubTool("_task_run", "Runs a task.", { id: { type: "integer" } }, ["id"]);
const taskRemoveStub = createStubTool("_task_remove", "Removes a task.", { id: { type: "integer" } }, ["id"]);
const taskPeekStub = createStubTool("_task_peek", "Gets task details.", { id: { type: "integer" } }, ["id"]);

// Register stubs...
toolRegistry.set(registerFunctionStub.name, registerFunctionStub);
toolRegistry.set(getFunctionCodeTool.name, getFunctionCodeTool); // Register the new implementation
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
            if (tool.inputSchema && Array.isArray(tool.inputSchema.required)) {
                for (const requiredParam of tool.inputSchema.required) {
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

// --- Global state variables ---
let cloudSocket: any;
let cloudReconnectTimer: NodeJS.Timeout | null = null;
let cloudConnectionAttempts: number = 0;

// --- New functions ---

const scheduleCloudReconnection = () => {
    // Check if we've reached the maximum attempts, ONLY if a limit is set (not null)
    if (CLOUD_MAX_RECONNECT_ATTEMPTS !== null && cloudConnectionAttempts >= CLOUD_MAX_RECONNECT_ATTEMPTS) {
        logger.error(`☁️❌ Reached maximum cloud reconnection attempts (${CLOUD_MAX_RECONNECT_ATTEMPTS}). Giving up.`);
        return; // Stop trying
    }
    if (cloudReconnectTimer) {
        clearTimeout(cloudReconnectTimer); // Clear existing timer if any
    }
    if (cloudConnectionAttempts >= CLOUD_MAX_RECONNECT_ATTEMPTS) {
        logger.error(`☁️❌ Reached max cloud reconnection attempts (${CLOUD_MAX_RECONNECT_ATTEMPTS}). Giving up.`);
        return;
    }

    cloudConnectionAttempts++;
    // Exponential backoff with jitter
    const delay = Math.min(
        CLOUD_MAX_RECONNECT_BACKOFF_MS,
        CLOUD_RECONNECT_DELAY_BASE_MS * Math.pow(2, cloudConnectionAttempts - 1)
    );
    const jitter = delay * 0.2 * Math.random(); // Add +/- 10% jitter
    const reconnectDelay = Math.round(delay + jitter);


    logger.info(`☁️ Scheduling cloud reconnection attempt ${cloudConnectionAttempts}/${CLOUD_MAX_RECONNECT_ATTEMPTS} in ${reconnectDelay}ms...`);
    cloudReconnectTimer = setTimeout(connectToCloud, reconnectDelay);
};

const connectToCloud = () => {
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
        clearTimeout(cloudReconnectTimer);
        cloudReconnectTimer = null;
    }

    // Construct the auth object matching server expectation
    const authPayload = { apiKey, email, serviceName, hostname: os.hostname() };
    logger.debug(`☁️ Auth payload being sent: ${JSON.stringify(authPayload)}`); // Stringify explicitly

    cloudSocket = io(`${cloudUrl}${CLOUD_NAMESPACE}`, { // Connect directly to the namespace URL
        path: '/socket.io', // Standard path, adjust if cloud server uses something else
        transports: ['websocket'], // Prefer websocket
        autoConnect: false, // We manage connection manually
        reconnection: false, // We manage reconnection manually
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
            logger.error("☁️ Authentication failed. Please check --email, --api-key, --service-name.");
            // Don't retry on auth failure
        } else {
            scheduleCloudReconnection();
        }
    });

    cloudSocket.on('disconnect', (reason: string) => { // Disconnect reason is a string
        logger.warn(`☁️🔌 Disconnected from cloud server. Reason: ${reason}`);
        cloudSocket = null; // Clear the socket instance
        // Reconnect unless intentionally disconnected or it was an auth error
        if (reason !== 'io server disconnect' && reason !== 'io client disconnect') {
             scheduleCloudReconnection();
        } else {
             logger.info(`☁️ Intentional disconnect from cloud. Won't reconnect automatically.`);
        }
    });

    // --- Placeholder for handling messages FROM the cloud ---
    // Listen for specific events the cloud server might emit
    cloudSocket.onAny(async (eventName: string, ...args: any[]) => { // Make callback async
        // Skip 'open', 'close', 'error' events for general logging if handled elsewhere
        if (['open', 'close', 'error', 'connect', 'connecting', 'reconnecting', 'disconnect'].includes(eventName)) return;

        // Log all events and their arguments clearly at INFO level by concatenating
        logger.debug(`☁️👇 Received cloud event '${eventName}': ${JSON.stringify(args)}`);

        // --- Handle specific messages from cloud ---
        if (eventName === 'service_message' && args.length > 0) {
            try {
                const request = args[0] as JsonRpcRequest;
                logger.info(`Processing JSON-RPC request: ${request.method} (ID: ${request.id})`);

                // Handle tools/list request
                if (request.method === 'tools/list') {
                    const tools = Array.from(toolRegistry.values()).map(tool => ({
                        name: tool.name,
                        description: tool.description,
                        inputSchema: tool.inputSchema
                    }));
                    const response: JsonRpcResponse = {
                        jsonrpc: '2.0',
                        result: { tools }, // According to MCP spec for tools/list
                        id: request.id
                    };
                    logger.info(`Responding to tools/list with ${tools.length} tools.`);
                    cloudSocket.emit('service_response', response); // Emit response event
                }
                // Handle tools/call request
                else if (request.method === 'tools/call') {
                    const { name, arguments: toolArgs } = request.params as { name: string, arguments: any }; // Extract tool name and args
                    const tool = toolRegistry.get(name);
                    let response: JsonRpcResponse; // Define response variable

                    if (tool) {
                        try {
                            // Basic required parameter check (TODO: Add proper jsonschema validation)
                            if (tool.inputSchema && Array.isArray(tool.inputSchema.required)) {
                                for (const requiredParam of tool.inputSchema.required) {
                                    if (!toolArgs || toolArgs[requiredParam] === undefined) {
                                        throw new Error(`Missing required parameter: '${requiredParam}'`);
                                    }
                                }
                            }

                            let resultContents: TextContent[];

                            if (tool.execute) { // Built-in or stub
                                logger.info(`Executing built-in/stub tool '${name}' with args: ${JSON.stringify(toolArgs)}`);
                                resultContents = await tool.execute(toolArgs); // Await built-in/stub execution
                            } else if (tool._function && tool.inputSchema) { // Dynamically loaded
                                logger.info(`Executing dynamic tool '${name}' from ${tool._filePath} with args: ${JSON.stringify(toolArgs)}`);

                                // Extract arguments based on inputSchema order (assuming required or properties keys)
                                // This is a simplification; a more robust solution might inspect parameter names/types
                                const paramNames = tool.inputSchema.required || Object.keys(tool.inputSchema.properties || {});
                                const orderedArgs = paramNames.map(paramName => {
                                    if (toolArgs[paramName] === undefined) {
                                        // This should have been caught by the required check earlier, but double-check
                                        throw new Error(`Internal error: Missing argument '${paramName}' for dynamic call, though required check passed.`);
                                    }
                                    return toolArgs[paramName];
                                });

                                // const dynamicResult = await tool._function(toolArgs); // Old call with object
                                const rawResult = await tool._function(...orderedArgs); // New call with spread arguments

                                // Format the raw result into MCP TextContent
                                // TODO: Handle cases where the rawResult might not be a number/string?
                                resultContents = [{ type: "text", text: rawResult.toString() }];
                            } else {
                                throw new Error(`Tool '${name}' found in registry but has no executable function or inputSchema.`);
                            }

                            // If execution reached here, it was successful
                            response = {
                                jsonrpc: '2.0',
                                result: { content: resultContents }, // MCP spec for tools/call result
                                id: request.id
                            };
                            logger.info(`Tool '${name}' executed successfully. Sending result.`);

                        } catch (error: any) {
                            // Handle validation errors OR execution errors
                            logger.error(`Error during 'tools/call' for '${name}': ${error.message}`);
                            if (error.stack) logger.debug(error.stack);
                            response = {
                                jsonrpc: '2.0',
                                // Use generic internal error code for now.
                                error: { code: -32000, message: `Tool execution failed: ${error.message}` },
                                id: request.id
                            };
                        }
                    } else {
                        // Tool not found
                        logger.error(`Tool '${name}' not found.`);
                        response = {
                            jsonrpc: '2.0',
                            error: { code: -32601, message: `Method not found: Tool '${name}'` }, // Method Not Found
                            id: request.id
                        };
                    }
                    // Send the response (success or error)
                    cloudSocket.emit('service_response', response);
                }
                // TODO: Handle other potential standard MCP methods (ping, initialize, etc.) if needed
                // else {
                //     logger.warn(`Received unhandled method: ${request.method}`);
                //     // Optionally send a MethodNotFound error
                // }
            } catch (jsonRpcError: any) {
                // Handle potential errors during JSON parsing or basic validation
                logger.error(`Failed to process incoming service_message: ${jsonRpcError.message}`);
                // Attempt to extract ID if possible, otherwise respond without ID
                let requestId: string | number | null = null;
                try {
                    const potentialRequest = JSON.parse(args[0].toString());
                    requestId = potentialRequest.id || null;
                } catch {}

                const errorResponse: JsonRpcResponse = {
                    jsonrpc: '2.0',
                    error: { code: -32700, message: `Parse error or invalid request: ${jsonRpcError.message}` }, // Parse Error
                    id: requestId
                };
                cloudSocket.emit('service_response', errorResponse);
            }
        }
        // --- Handle other cloud events if necessary ---
        // else if (eventName === 'some_other_event') { ... }

    }); // End cloudSocket.onAny
    // --- End Placeholder ---

    cloudSocket.on('error', (error: Error) => {
        logger.error(`☁️ Socket.IO general error: ${error.message}`);
        // General errors might not trigger disconnect, schedule check/reconnect
        if (!cloudSocket?.connected) {
             scheduleCloudReconnection();
        }
    });

    // Manually initiate the connection
    cloudSocket.connect();
};

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

    // --- Connect to Cloud Server ---
    connectToCloud(); // Initiate connection after local server is up
};

// --- Function to Check Existing Process ---
function checkAndHandleExistingProcess(): void {
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

// --- Graceful Shutdown ---
const shutdown = async (): Promise<void> => {
    let exitCode = 0; // Default to success
    logger.info("🚦 Shutting down gracefully...");
    try {
        // Main shutdown operations
        server.close(() => logger.info('🛑 HTTP server closed.'));
        logger.info(`🔌 Closing ${wss.clients.size} client connections...`);
        wss.clients.forEach(client => { if (client.readyState === WebSocket.OPEN) client.close(); });

        // --- Disconnect from Cloud ---
        if (cloudReconnectTimer) {
            clearTimeout(cloudReconnectTimer); // Clear existing timer if any
            cloudReconnectTimer = null;
            logger.info("☁️ Cleared pending cloud reconnect timer.");
        }
        if (cloudSocket && cloudSocket.connected) {
            logger.info("☁️ Disconnecting from cloud server...");
            cloudSocket.disconnect(); // Trigger intentional disconnect
            cloudSocket = null;
        }

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
// First, check if already running
checkAndHandleExistingProcess();

// If the check didn't exit, start the server
startServer().catch(error => {
    logger.error(`🚨 Failed to start server: ${error.message}`, { stack: error.stack });
    process.exit(1);
});
