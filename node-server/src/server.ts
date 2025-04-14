import WebSocket, { WebSocketServer } from 'ws';
import http from 'http';
import url from 'url';
import winston from 'winston';
import path from 'path';
import fs from 'fs/promises';
import { existsSync, readFileSync, writeFileSync, unlinkSync, statSync } from 'fs';
import io from 'socket.io-client'; // Standard import for the function
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import os from 'os'; // ADDED IMPORT
import process from 'process'; // Explicit import for process.kill
import * as ts from 'typescript'; // Added for Compiler API
import { checkAndHandleExistingProcess } from './util'; // <-- RE-ADDED IMPORT

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
    level: argv['log-level'], // Use yargs default directly
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
const CLOUD_MAX_RECONNECT_ATTEMPTS: number | null = null;
const CLOUD_MAX_RECONNECT_BACKOFF_MS = 60000; // 60 seconds

// --- MCP Tool Registry ---
const toolRegistry = new Map<string, ToolDefinition>();
// Keep track of file path per dynamic function name (using compiled path)
const dynamicFunctionFiles = new Map<string, string>();

// --- Task Management --- //
const tasks = new Map<number, any>(); // Stores task payloads keyed by ID
let nextTaskId = 1; // Counter for assigning new task IDs

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

// --- Helper Function for Formatting Tool List ---
const prepareToolsListPayload = (): any[] => {
    return Array.from(toolRegistry.values()).map(tool => {
        // Destructure, keeping _filePath for now
        const { _function, _filePath, execute, ...baseToolDefinition } = tool;

        // Create a mutable object to potentially add the timestamp
        let toolToSend: any = { ...baseToolDefinition };

        // If it's a dynamic function with a path, get and add the timestamp
        if (_filePath && typeof _filePath === 'string') {
            try {
                const stats = statSync(_filePath); // Use statSync
                toolToSend.lastUpdated = stats.mtime.toISOString(); // Use 'lastUpdated' field name
                logger.debug(`Added lastUpdated timestamp for ${_filePath}`); // Update log message too
            } catch (statError: any) {
                logger.warn(`Could not get stats for dynamic function file ${_filePath}: ${statError.message}`);
                // Proceed without timestamp if stat fails
            }
        }
        logger.debug(`Tool definition being returned:\n${JSON.stringify(toolToSend, null, 2)}`); // Combine into one message arg
        return toolToSend; // Return the potentially modified definition
    });
};

// --- Dynamic Function Loading ---
const loadAndRegisterDynamicFunction = async (compiledFilePath: string): Promise<void> => {
    // Use compiled JS file path
    const functionName = path.basename(compiledFilePath, '.js');

    // Check if a tool with this name ALREADY exists and is NOT tracked as a dynamic file
    // (meaning it's likely a built-in or manually registered tool, which we shouldn't overwrite on startup scan)
    if (toolRegistry.has(functionName) && !dynamicFunctionFiles.has(functionName)) {
        logger.warn(`⚠️ Skipping dynamic load for '${functionName}': name conflicts with a non-dynamic tool.`);
        return;
    }

    logger.debug(`Attempting to load dynamic function '${functionName}' from ${compiledFilePath}`);

    try {
        // Get file stats to log modification time
        const stats = await fs.stat(compiledFilePath);

        // Log file's last modified timestamp before loading
        logger.info(`⏳ Loading dynamic function '${functionName}' (modified: ${stats.mtime.toISOString()}) from ${compiledFilePath}`);

        // Ensure cache is clear for the specific file being loaded/reloaded
        const absolutePath = require.resolve(compiledFilePath);
        delete require.cache[absolutePath];
        logger.debug(`Cleared require cache for: ${absolutePath}`)

        const dynamicModule: DynamicFunctionModule = require(absolutePath);

        // Validate Module Structure (using types)
        if (typeof dynamicModule.handler !== 'function') throw new Error(`Module does not export a 'handler' function.`);
        if (typeof dynamicModule.metadata !== 'object' || dynamicModule.metadata === null) {
            // If no metadata object, create a default one
            dynamicModule.metadata = { description: `Dynamic function: ${functionName}` };
            logger.debug(`No metadata found for ${functionName}, using default.`);
        }

        // Use default description if none provided
        if (typeof dynamicModule.metadata.description !== 'string' || !dynamicModule.metadata.description) {
            dynamicModule.metadata.description = `Dynamic function: ${functionName}`;
            logger.debug(`No description found for ${functionName}, using default.`);
        }

        // Try to create a schema from source if no explicit schema provided
        let inputSchema = dynamicModule.metadata.inputSchema;

        if (!inputSchema) {
            // Look for the TypeScript source file
            const tsFilePath = path.join(SOURCE_FUNCTIONS_DIR, `${functionName}.ts`);

            if (existsSync(tsFilePath)) {
                try {
                    // Read the TypeScript source file
                    const sourceCode = readFileSync(tsFilePath, 'utf8');
                    inputSchema = extractSchemaFromTypeScript(sourceCode, functionName);
                    logger.debug(`Generated schema for '${functionName}' from TypeScript source: ${JSON.stringify(inputSchema)}`);
                } catch (parseError: any) {
                    logger.warn(`Failed to extract schema from TypeScript for '${functionName}': ${parseError.message}`);
                    // Fall back to empty schema
                    inputSchema = { type: 'object', properties: {} };
                }
            } else {
                logger.warn(`Could not find TypeScript source for '${functionName}' at ${tsFilePath}`);
                inputSchema = { type: 'object', properties: {} };
            }
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

// Helper function to extract parameter schema from TypeScript source
const extractSchemaFromTypeScript = (sourceCode: string, functionName: string): any => {
    const properties: Record<string, any> = {};
    const required: string[] = [];

    try {
        // Parse the TypeScript source file
        const sourceFile = ts.createSourceFile(
            'temp.ts',
            sourceCode,
            ts.ScriptTarget.Latest,
            true
        );

        // Find the handler function in the source
        let handlerFunction: ts.FunctionDeclaration | undefined;

        // Look for exported handler function or any function with functionName
        ts.forEachChild(sourceFile, node => {
            // Check for export const handler = function or export const handler = (params) => {}
            if (ts.isVariableStatement(node) &&
                node.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword)) {

                node.declarationList.declarations.forEach(declaration => {
                    if (ts.isIdentifier(declaration.name) &&
                        declaration.name.text === 'handler' &&
                        declaration.initializer) {

                        // Found handler as variable, now check if it's a function
                        if (ts.isFunctionExpression(declaration.initializer) ||
                            ts.isArrowFunction(declaration.initializer)) {
                            // Handle params from this function expression
                            extractParamsFromFunction(declaration.initializer, properties, required);
                        }
                    }
                });
            }

            // Check for direct function declarations (such as "export function functionName")
            if (ts.isFunctionDeclaration(node) &&
                node.name &&
                node.name.text === functionName &&
                node.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword)) {
                handlerFunction = node;
            }
        });

        // If we found a direct function declaration, extract params
        if (handlerFunction && handlerFunction.parameters) {
            extractParamsFromFunction(handlerFunction, properties, required);
        }

        // Build the schema object
        const schema: any = {
            type: 'object',
            properties: properties
        };

        if (required.length > 0) {
            schema.required = required;
        }

        return schema;
    } catch (e: any) {
        logger.warn(`Error extracting schema from TypeScript: ${e.message}`);
        return { type: 'object', properties: {} };
    }
};

// Helper to extract parameters from a function
const extractParamsFromFunction = (node: ts.FunctionLikeDeclaration,
                                 properties: Record<string, any>,
                                 required: string[]) => {
    // Process each parameter
    node.parameters.forEach(param => {
        // Get parameter name
        if (ts.isIdentifier(param.name)) {
            const paramName = param.name.text;
            let paramType = 'any';

            // Try to determine parameter type
            if (param.type) {
                if (ts.isTypeReferenceNode(param.type)) {
                    if (ts.isIdentifier(param.type.typeName)) {
                        const typeName = param.type.typeName.text;

                        // Map TypeScript types to JSON Schema types
                        switch (typeName.toLowerCase()) {
                            case 'string': paramType = 'string'; break;
                            case 'number': paramType = 'number'; break;
                            case 'boolean': paramType = 'boolean'; break;
                            case 'array': paramType = 'array'; break;
                            case 'object': paramType = 'object'; break;
                            default: paramType = 'any';
                        }
                    }
                } else {
                    // Check for keyword types by examining the node kind directly
                    switch (param.type.kind) {
                        case ts.SyntaxKind.StringKeyword: paramType = 'string'; break;
                        case ts.SyntaxKind.NumberKeyword: paramType = 'number'; break;
                        case ts.SyntaxKind.BooleanKeyword: paramType = 'boolean'; break;
                        default: paramType = 'any';
                    }
                }
            }

            // Add to properties
            properties[paramName] = {
                type: paramType,
                description: `Parameter '${paramName}'`
            };

            // If no initializer (default value), mark as required
            if (!param.initializer && !param.questionToken) {
                required.push(paramName);
            }
        }
    });
};

// --- NEW Helper Function: Inspect TypeScript Function --- //
async function inspectTypeScriptFunction(code: string, exportedFunctionName: string): Promise<{ description: string, inputSchema: any }> {
    logger.debug(`Inspecting TS code for function: ${exportedFunctionName}`);
    let description = `Dynamically registered function: ${exportedFunctionName}`; // Default description
    const properties: Record<string, any> = {};
    const required: string[] = [];

    try {
        const sourceFile = ts.createSourceFile(
            'tempInspect.ts', // Temporary name for parsing
            code,
            ts.ScriptTarget.Latest,
            true // setParentNodes
        );

        let foundFunctionNode: ts.FunctionDeclaration | ts.FunctionExpression | ts.ArrowFunction | null = null;

        // Find the specific exported function node
        ts.forEachChild(sourceFile, node => {
            if (foundFunctionNode) return; // Stop if found

            // Exported Function Declaration (export function name() { ... })
            if (ts.isFunctionDeclaration(node) && node.name && node.name.getText(sourceFile) === exportedFunctionName &&
                node.modifiers?.some(mod => mod.kind === ts.SyntaxKind.ExportKeyword)) {
                foundFunctionNode = node;
            }
            // Exported Variable Statement (export const name = () => { ... } or export const name = function() { ... })
            else if (ts.isVariableStatement(node) && node.modifiers?.some(mod => mod.kind === ts.SyntaxKind.ExportKeyword)) {
                node.declarationList.declarations.forEach(declaration => {
                    if (foundFunctionNode) return;
                    if (ts.isIdentifier(declaration.name) && declaration.name.getText(sourceFile) === exportedFunctionName &&
                        declaration.initializer && (ts.isArrowFunction(declaration.initializer) || ts.isFunctionExpression(declaration.initializer))) {
                        foundFunctionNode = declaration.initializer; // Get the function expression/arrow function itself
                    }
                });
            }
        });

        if (!foundFunctionNode) {
            logger.warn(`Could not find AST node for exported function '${exportedFunctionName}' during inspection.`);
            // Proceed with default description and empty schema
        } else {
            logger.debug(`Found AST node for '${exportedFunctionName}'`);

            // 1. Extract Description from JSDoc comments attached to the *parent* node if possible
            let commentNode: ts.Node = foundFunctionNode;
            // If it's an initializer, try getting comments from the variable declaration or statement
            if ( (ts.isArrowFunction(foundFunctionNode) || ts.isFunctionExpression(foundFunctionNode) ) && foundFunctionNode.parent && ts.isVariableDeclaration(foundFunctionNode.parent)) {
                 commentNode = foundFunctionNode.parent; // Check VariableDeclaration first
                 if (commentNode.parent && ts.isVariableDeclarationList(commentNode.parent) && commentNode.parent.parent && ts.isVariableStatement(commentNode.parent.parent)){
                     commentNode = commentNode.parent.parent; // Then check VariableStatement
                 }
            } else if (ts.isFunctionDeclaration(foundFunctionNode)) {
                 commentNode = foundFunctionNode; // Check the function declaration itself
            }

            // Use ts.getJSDocCommentsAndTags - this handles multiline comments better
             const comments = ts.getJSDocCommentsAndTags(commentNode);
             if (comments.length > 0 && comments[0].comment) {
                if (typeof comments[0].comment === 'string') {
                    description = comments[0].comment.trim();
                } else if (Array.isArray(comments[0].comment)) {
                    // Handle NodeArray<JSDocComment>
                     description = comments[0].comment.map(c => c.text).join('\\n').trim();
                 }
                logger.debug(`Extracted description: "${description}"`);
            }


            // 2. Extract Parameters for Schema
            if (foundFunctionNode.parameters) {
                foundFunctionNode.parameters.forEach(param => {
                    // Need to handle parameter names correctly (Identifier, BindingPattern)
                    if (ts.isIdentifier(param.name)) {
                        const paramName = param.name.getText(sourceFile);
                        const propDetails: any = {};

                        // Determine Type
                        let paramType = 'any'; // Default
                        if (param.type) {
                            switch (param.type.kind) {
                                case ts.SyntaxKind.StringKeyword: paramType = 'string'; break;
                                case ts.SyntaxKind.NumberKeyword: paramType = 'number'; break;
                                case ts.SyntaxKind.BooleanKeyword: paramType = 'boolean'; break;
                                case ts.SyntaxKind.ObjectKeyword: paramType = 'object'; break; // Could potentially refine
                                case ts.SyntaxKind.ArrayType: paramType = 'array'; break;
                                case ts.SyntaxKind.AnyKeyword: paramType = 'any'; break;
                                case ts.SyntaxKind.TypeReference:
                                     // Handle basic type references if needed (e.g., 'Promise<string>' might just be 'string' for schema?)
                                     // For simplicity now, treat complex references as 'any' or 'object'
                                     logger.debug(`Parameter '${paramName}' has complex TypeReference: ${param.type.getText(sourceFile)}, using 'any' for schema.`);
                                     paramType = 'any';
                                     break;
                                default:
                                     logger.debug(`Parameter '${paramName}' has unhandled type kind: ${ts.SyntaxKind[param.type.kind]}, using 'any' for schema.`);
                                     paramType = 'any';
                                     break;
                            }
                        } else {
                             logger.debug(`Parameter '${paramName}' has no explicit type, using 'any' for schema.`);
                        }
                        propDetails.type = paramType;

                        // Add description from JSDoc @param tag if available (requires more complex JSDoc parsing)
                        // Find @param tag for this parameter name in comments extracted earlier
                        // ... implementation needed ...
                        // propDetails.description = extractedParamDescription;

                        properties[paramName] = propDetails;
                        logger.debug(`Found param: ${paramName}, schema type: ${paramType}`);

                        // Determine if required (no '?' token AND no initializer)
                        if (!param.questionToken && !param.initializer) {
                            required.push(paramName);
                            logger.debug(`Param '${paramName}' is required.`);
                        } else {
                             logger.debug(`Param '${paramName}' is optional.`);
                        }
                    } else {
                         logger.warn(`Found complex parameter structure (not Identifier) for parameter in ${exportedFunctionName}, skipping for schema generation.`);
                         // Handling destructuring parameters ({ a, b }: { a: string, b: number }) is complex
                    }
                });
            } else {
                 logger.debug(`Function '${exportedFunctionName}' has no parameters.`);
            }
        }
    } catch (inspectError: any) {
        logger.error(`Error inspecting TypeScript code for schema/description: ${inspectError.message}`, { stack: inspectError.stack });
        // Fallback to defaults
        description = `Dynamically registered function: ${exportedFunctionName}`;
        Object.keys(properties).forEach(key => delete properties[key]); // Clear properties
        required.length = 0; // Clear required
    }

    const inputSchema = {
        type: "object",
        properties: properties,
        ...(required.length > 0 && { required: required }) // Only add required if it's not empty
    };

    logger.debug(`Generated Schema: ${JSON.stringify(inputSchema)}`);
    logger.debug(`Using Description: "${description}"`);

    return { description, inputSchema };
}


// --- Stub Tool Definitions ---
const createStubTool = (
    name: string,
    description: string,
    inputSchema: { [key: string]: ToolParameterProperty },
    requiredParams: string[] = []
): ToolDefinition => {
    return {
        name: name,
        description: description,
        inputSchema: { type: "object", properties: inputSchema, required: requiredParams },
        async execute(args: any): Promise<TextContent[]> {
            logger.info(`Executing STUB for tool: ${name} with args: ${JSON.stringify(args)}`);
            return [{ type: "text", text: `Tool '${name}' called successfully (stub implementation).` }];
        }
    };
};

// Define stubs... (keep these as before)
// const registerFunctionStub = createStubTool("_function_register", "Registers a new dynamic function.", { function_name: { type: "string" }, code: { type: "string" } }, ["function_name", "code"]); // Replaced with full implementation below

// --- Custom implementation for _function_register ---
const registerFunctionTool: ToolDefinition = {
    name: "_function_register",
    description: "Sets the content of a dynamic TypeScript function",
    inputSchema: {
        type: "object",
        properties: {
            code: { type: "string", description: "The TypeScript source code containing a single function definition." }
        },
        required: ["code"]
    },
    // Make 'code' non-optional as it's required by the schema
    async execute(args: { code: string }): Promise<TextContent[]> {
        const code = args.code;

        if (!code) {
            throw new Error("Missing required argument: code");
        }

        // --- Extract function name from code --- PRE-SAVE STEP
        let functionName: string | null = null;
        // We need the functionName early to determine the filename

        try {
            const tempSourceFile = ts.createSourceFile('tempExtract.ts', code, ts.ScriptTarget.Latest, true);
            ts.forEachChild(tempSourceFile, node => {
                if (functionName) return; // Stop searching once found

                // Case 1: Exported Function Declaration
                if (ts.isFunctionDeclaration(node) && node.name && node.modifiers?.some(mod => mod.kind === ts.SyntaxKind.ExportKeyword)) {
                    functionName = node.name.getText(tempSourceFile);
                    logger.debug(`Found exported function declaration: ${functionName}`);
                }
                // Case 2: Exported Variable Statement (assigned to a function)
                else if (ts.isVariableStatement(node) && node.modifiers?.some(mod => mod.kind === ts.SyntaxKind.ExportKeyword)) {
                    node.declarationList.declarations.forEach(declaration => {
                        if (functionName) return;
                        if (ts.isIdentifier(declaration.name) && declaration.initializer &&
                            (ts.isArrowFunction(declaration.initializer) || ts.isFunctionExpression(declaration.initializer))) {
                            functionName = declaration.name.getText(tempSourceFile);
                            logger.debug(`Found exported variable assigned to a function: ${functionName}`);
                        }
                    });
                }
            });

            if (!functionName) {
                throw new Error("Could not find an exported function declaration or an exported variable assigned to a function in the provided code.");
            }
            logger.info(`Extracted function name from code: ${functionName}`);
        } catch (parseError: any) {
            logger.error(`Failed to parse code or extract function name: ${parseError.message}`);
            throw new Error(`Failed to parse code: ${parseError.message}`);
        }
        // --- End extraction ---

        // Basic validation for extracted function name
        if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(functionName)) {
            throw new Error(`Invalid extracted function name '${functionName}': Must start with a letter or underscore, followed by letters, numbers, or underscores.`);
        }

        const sourceFilePath = path.join(SOURCE_FUNCTIONS_DIR, `${functionName}.ts`);
        const compiledFilePath = path.join(COMPILED_FUNCTIONS_DIR, `${functionName}.js`);

        logger.info(`Attempting to register function '${functionName}' (name derived from code)...`);

        // 1. Save the TypeScript code
        try {
            await fs.mkdir(SOURCE_FUNCTIONS_DIR, { recursive: true });
            await fs.writeFile(sourceFilePath, code, 'utf8');
            logger.info(`Saved TypeScript source to: ${sourceFilePath}`);
        } catch (writeError: any) {
            logger.error(`Failed to write source file ${sourceFilePath}: ${writeError.message}`);
            throw new Error(`Failed to save function source: ${writeError.message}`);
        }

        // 2. Compile the TypeScript file
        try {
            await fs.mkdir(COMPILED_FUNCTIONS_DIR, { recursive: true });

            const compilerOptions: ts.CompilerOptions = {
                outDir: COMPILED_FUNCTIONS_DIR,
                target: ts.ScriptTarget.ES2016, // Or choose appropriate target
                module: ts.ModuleKind.CommonJS,
                esModuleInterop: true,
                skipLibCheck: true,
                resolveJsonModule: true,
                declaration: false, // Don't need .d.ts files for dynamic execution
            };

            logger.info(`Compiling ${sourceFilePath} using TypeScript API...`);
            const program = ts.createProgram([sourceFilePath], compilerOptions);
            const emitResult = program.emit();

            const allDiagnostics = ts.getPreEmitDiagnostics(program).concat(emitResult.diagnostics);
            let hasError = false;
            const diagnosticMessages: string[] = [];

            allDiagnostics.forEach(diagnostic => {
                const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
                if (diagnostic.file) {
                    const { line, character } = ts.getLineAndCharacterOfPosition(diagnostic.file, diagnostic.start!);
                    diagnosticMessages.push(`${path.basename(diagnostic.file.fileName)} (${line + 1},${character + 1}): ${message}`);
                } else {
                    diagnosticMessages.push(message);
                }
                if (diagnostic.category === ts.DiagnosticCategory.Error) {
                    hasError = true;
                }
            });

            if (hasError || emitResult.emitSkipped) {
                const errorOutput = diagnosticMessages.join('\n');
                logger.error(`Compilation failed for ${functionName}. Diagnostics:\n${errorOutput}`);
                try { await fs.unlink(sourceFilePath); } catch { /* Ignore cleanup error */ }
                throw new Error(`Compilation failed:\n${errorOutput}`);
            }

            logger.info(`Successfully compiled ${functionName} to: ${compiledFilePath}`);

        } catch (compileError: any) {
             logger.error(`Error during compilation step for ${functionName}: ${compileError.message}`, { stack: compileError.stack });
             // Ensure source file is cleaned up if compilation setup failed too
             try { if(existsSync(sourceFilePath)) await fs.unlink(sourceFilePath); } catch { /* Ignore */ }
             throw compileError; // Re-throw the error
        }

        // 3. Load, Inspect, and Register
        let userFunction: Function | null = null;
        let userFunctionName: string | null = null; // To store the name found within the module (e.g., 'default' or named export)
        try {
            // Ensure require cache is clear before loading the new module
            const absolutePath = require.resolve(compiledFilePath);
            delete require.cache[absolutePath];
            logger.debug(`Cleared require cache for compiled file: ${absolutePath}`)

            const dynamicModule = require(absolutePath);

            // Find the single exported function (named or default)
            const exports = Object.keys(dynamicModule);
            const potentialFunctions = exports.filter(key => typeof dynamicModule[key] === 'function' && key !== '__esModule');

            if (potentialFunctions.length === 0) {
                if (dynamicModule.default && typeof dynamicModule.default === 'function') {
                    logger.info(`Found default export function in module ${functionName}.js`);
                    userFunction = dynamicModule.default;
                    userFunctionName = 'default';
                } else {
                    throw new Error(`Compiled module ${functionName}.js does not export any functions (checked named and default).`);
                }
            } else if (potentialFunctions.length > 1) {
                if (dynamicModule.default && typeof dynamicModule.default === 'function') {
                    throw new Error(`Compiled module ${functionName}.js exports multiple named functions (${potentialFunctions.join(', ')}) AND a default export. Please export only ONE function.`);
                } else {
                    throw new Error(`Compiled module ${functionName}.js exports multiple named functions (${potentialFunctions.join(', ')}). Please export only one function.`);
                }
            } else {
                userFunctionName = potentialFunctions[0];
                userFunction = dynamicModule[userFunctionName];
                logger.info(`Found single exported function '${userFunctionName}' in module ${functionName}.js`);
            }

            if (!userFunction) { // Should be unreachable
                 throw new Error(`Could not locate the function to execute within ${functionName}.js`);
            }

            // --- Inspect original TS code for Schema and Description ---
            const { description: generatedDescription, inputSchema: generatedInputSchema } = await inspectTypeScriptFunction(code, functionName);
            // --- Inspection Complete ---

            // Construct the ToolDefinition
            const constructedTool: ToolDefinition = {
                name: functionName, // Use the name derived from code (for the tool registry key)
                description: generatedDescription,
                inputSchema: generatedInputSchema,
                async execute(toolArgs: any): Promise<TextContent[]> {
                    const actualUserFuncName = userFunctionName || 'default'; // Use the name found in the module
                    logger.info(`Executing dynamic tool '${functionName}' (wraps '${actualUserFuncName}') with args: ${JSON.stringify(toolArgs)}`);

                    let orderedArgs: any[] = [];
                    const currentToolArgs = (toolArgs && typeof toolArgs === 'object') ? toolArgs : {};

                    if (generatedInputSchema && generatedInputSchema.properties && typeof generatedInputSchema.properties === 'object') {
                        const argNames = Object.keys(generatedInputSchema.properties);
                        orderedArgs = argNames.map(argName => {
                            if (currentToolArgs.hasOwnProperty(argName)) {
                                return currentToolArgs[argName];
                            } else {
                                const isRequired = generatedInputSchema.required && generatedInputSchema.required.includes(argName);
                                if (isRequired) {
                                    logger.error(`Required argument '${argName}' missing for tool '${functionName}'.`);
                                    throw new Error(`Required argument '${argName}' is missing.`);
                                } else {
                                    logger.debug(`Optional argument '${argName}' not provided for tool '${functionName}', passing undefined.`);
                                    return undefined;
                                }
                            }
                        });
                        logger.debug(`Prepared ordered arguments based on generated schema: ${orderedArgs.length} args`);
                    } else {
                        logger.debug(`Generated schema has no properties for tool '${functionName}', calling function without arguments.`);
                        orderedArgs = [];
                    }

                    // Call the actual user function loaded from the compiled JS
                    try {
                        const result = await (userFunction as (...args: any[]) => Promise<any>)(...orderedArgs);

                        // Format result
                        let resultString: string;
                         try {
                            if (result === undefined || result === null) {
                                resultString = "";
                            } else if (typeof result === 'string') {
                                resultString = result;
                            } else { // Objects, numbers, booleans, arrays etc.
                                resultString = JSON.stringify(result, null, 2);
                            }
                        } catch (stringifyError: any) {
                            logger.warn(`Failed to stringify result for tool '${functionName}': ${stringifyError.message}. Returning raw string representation.`);
                            resultString = String(result);
                        }

                        logger.info(`Dynamic tool '${functionName}' executed successfully.`);
                        return [{ type: "text", text: resultString }];
                    } catch (executionError: any) {
                         logger.error(`Error executing dynamic tool '${functionName}' (function '${actualUserFuncName}'): ${executionError.message}`, { stack: executionError.stack });
                        throw new Error(`Execution error in tool '${functionName}': ${executionError.message}`);
                    }
                } // End of inner execute
            }; // End of constructedTool definition

            // --- Register the tool ---
            if (toolRegistry.has(functionName)) {
                logger.warn(`⚠️ Overwriting existing tool registration for '${functionName}'`);
                // You might want to remove the old files here if overwriting
                // const oldCompiledPath = dynamicFunctionFiles.get(functionName); ... remove old files ...
            }
            toolRegistry.set(functionName, constructedTool);
            dynamicFunctionFiles.set(functionName, compiledFilePath); // Track the *compiled* path
            logger.info(`✅ Successfully registered dynamic tool: ${functionName}`);

            // Update cloud if connected
            if (cloudSocket && cloudSocket.connected) {
                logger.info(`☁️ Emitting tools_updated to cloud due to registration of ${functionName}`);
                cloudSocket.emit('tools_updated', { tools: prepareToolsListPayload() });
            }

            return [{ type: "text", text: `Function '${functionName}' registered successfully.` }];

        } catch (loadError: any) {
            logger.error(`Failed to load or register compiled function ${functionName}: ${loadError.message}`, {stack: loadError.stack});
            // Attempt cleanup of compiled/source files if loading/registration failed
            try { if(existsSync(compiledFilePath)) await fs.unlink(compiledFilePath); } catch { /* Ignore */ }
            try { if(existsSync(sourceFilePath)) await fs.unlink(sourceFilePath); } catch { /* Ignore */ }
            throw new Error(`Failed to load/register function: ${loadError.message}`);
        }
    } // End of outer execute method
};

// --- Custom implementation for _function_get ---
const getFunctionCodeTool: ToolDefinition = {
    name: "_function_get",
    description: "Gets the TypeScript source code for a dynamic function",
    inputSchema: {
        type: "object",
        properties: {
            name: { type: "string", description: "The name of the function (without .ts extension)" } // Changed from function_name
        },
        required: ["name"] // Changed from function_name
    },
    // Make 'name' non-optional
    async execute(args: { name: string }): Promise<TextContent[]> { // Changed param name in type
        const functionName = args.name; // Use 'name'
        if (!functionName) {
            throw new Error("Missing required argument for _function_get: name");
        }

        logger.info(`📄 GETTING CODE AND DESC FOR FUNCTION: ${functionName}`);

        // Construct path to the *source* TypeScript file
        const sourceFilePath = path.join(SOURCE_FUNCTIONS_DIR, `${functionName}.ts`);

        // Check if the source file exists
        if (!existsSync(sourceFilePath)) {
            throw new Error(`Function source file '${functionName}.ts' not found in ${SOURCE_FUNCTIONS_DIR}.`);
        }

        try {
            // Read the source code
            const code = await fs.readFile(sourceFilePath, 'utf-8');

            // TODO: Extract description from comments/docstring in the TS file if needed
            // For now, provide a default description
            const description = "Dynamically registered TypeScript function"; // Placeholder

            // Prepare the result data matching Python's structure
            const resultData = {
                name: functionName,
                code: code,
                description: description
            };

            logger.info(`✅ SUCCESSFULLY RETRIEVED CODE AND DESC FOR: ${functionName}`);
            // Return the result as a JSON string in the text field
            return [{ type: "text", text: JSON.stringify(resultData) }];
        } catch (error: any) {
            logger.error(`❌ ERROR READING FUNCTION FILE ${sourceFilePath}: ${error.message}`);
            throw new Error(`Failed to read function source for '${functionName}': ${error.message}`);
        }
    }
};

// --- Custom implementation for _function_remove ---
const removeFunctionTool: ToolDefinition = {
    name: "_function_remove",
    description: "Removes a dynamic Typescript function",
    inputSchema: {
        type: "object",
        properties: {
            name: { type: "string", description: "The name of the function to remove" }
        },
        required: ["name"]
    },
    // Make 'name' non-optional
    async execute(args: { name: string }): Promise<TextContent[]> {
        const functionName = args.name;
        if (!functionName) {
            throw new Error("Missing required argument: name");
        }

        logger.info(`Attempting to remove function '${functionName}'...`);

        const sourceFilePath = path.join(SOURCE_FUNCTIONS_DIR, `${functionName}.ts`);
        const compiledFilePath = path.join(COMPILED_FUNCTIONS_DIR, `${functionName}.js`);

        let wasRegistered = false;
        let tsDeleted = false;
        let jsDeleted = false;

        // 1. Unregister from Tool Registry
        if (toolRegistry.has(functionName)) {
            toolRegistry.delete(functionName);
            wasRegistered = true;
            logger.info(`Unregistered function '${functionName}' from tool registry.`);
        } else {
             // If it wasn't registered, still proceed to delete files, but log it.
             logger.warn(`Function '${functionName}' was not found in the tool registry. Attempting file cleanup anyway.`);
        }

        // 2. Delete TypeScript source file
        try {
            await fs.rm(sourceFilePath);
            tsDeleted = true;
            logger.info(`Deleted source file: ${sourceFilePath}`);
        } catch (error: any) {
            if (error.code === 'ENOENT') {
                logger.info(`Source file not found (already deleted?): ${sourceFilePath}`);
            } else {
                logger.error(`Failed to delete source file ${sourceFilePath}: ${error.message}`);
                // Decide if this is critical enough to throw? For cleanup, maybe not.
            }
        }

        // 3. Delete JavaScript compiled file
        try {
            await fs.rm(compiledFilePath);
            jsDeleted = true;
            logger.info(`Deleted compiled file: ${compiledFilePath}`);
        } catch (error: any) {
            if (error.code === 'ENOENT') {
                logger.info(`Compiled file not found (already deleted?): ${compiledFilePath}`);
            } else {
                logger.error(`Failed to delete compiled file ${compiledFilePath}: ${error.message}`);
            }
        }

        let message = `Function '${functionName}' removal process finished.`;
        if (wasRegistered) message += " Unregistered from tools.";
        if (tsDeleted) message += " Source file deleted.";
        if (jsDeleted) message += " Compiled file deleted.";
        if (!wasRegistered && !tsDeleted && !jsDeleted) message = `Function '${functionName}' not found (neither in registry nor as files).`

        return [{ type: "text", text: message }];
    }
};

// --- NEW Tool: Add Placeholder Function --- //
const addFunctionTool: ToolDefinition = {
    name: "_function_add",
    description: "Adds a new, empty placeholder TypeScript function with the given name.",
    inputSchema: {
        type: "object",
        properties: {
            name: { type: "string", description: "The name to register the new placeholder function under." }
        },
        required: ["name"]
    },
    // Make 'name' non-optional
    async execute(args: { name: string }): Promise<TextContent[]> {
        const name = args.name;

        if (!name) {
            throw new Error("Missing required argument: name");
        }

        logger.info(`➕ ADDING NEW PLACEHOLDER FUNCTION: ${name}`);

        // Define the placeholder content using the provided name
        const placeholderCode = `// A description of the function
export function ${name}(): string {
    console.log(\`Executing placeholder function '${name}'\`);
    // Add your logic here!
    return "Placeholder function executed successfully.";
}`;

        // Prepare arguments for the existing _function_register method
        // We now only need name and code - description and schema are optional
        const registrationArgs: { name: string, code: string } = {
            name: name, // The name provided by the user for registration
            code: placeholderCode
        };

        try {
            // Call the existing registration logic
            logger.debug(`Calling internal _function_register for placeholder '${name}'...`);
            const result = await registerFunctionTool.execute(registrationArgs);
            logger.info(`✅ Successfully added placeholder function: ${name}`);

            // Modify the success message if possible
            if (result && result.length > 0 && result[0].type === 'text') {
                 const originalMessage = result[0].text;
                 result[0].text = `${originalMessage} (This is a placeholder, edit dynamic_functions/ts/${name}.ts to implement logic).`;
            }
            return result;
        } catch (e: any) {
            logger.error(`❌ ERROR ADDING PLACEHOLDER FUNCTION ${name}: ${e.message}`);
            // _function_register should handle cleanup, just re-raise
            throw e;
        }
    }
};

// --- NEW Tool: Add Task --- //
const addTaskTool: ToolDefinition = {
    name: "_task_add",
    description: "Adds a new task",
    inputSchema: {
        type: "object",
        properties: {
            payload: { type: "object", description: "The JSON object containing the task details." }
        },
        required: ["payload"]
    },
    // Make 'payload' non-optional
    async execute(args: { payload: any }): Promise<TextContent[]> {
        logger.info(`⚙️ TASK ADD CALLED with args: ${JSON.stringify(args)}`);
        try {
            const taskPayload = args.payload;
            if (!taskPayload) {
                throw new Error("Missing 'payload' in arguments");
            }
            // Basic check if it's an object (might need deeper validation depending on requirements)
            if (typeof taskPayload !== 'object' || taskPayload === null || Array.isArray(taskPayload)) {
                 throw new Error("'payload' must be a JSON object (dictionary)");
            }

            // Generate a new task ID
            const taskId = nextTaskId++;

            // Store the task details
            tasks.set(taskId, taskPayload);

            logger.info(`✅ Task added with ID: ${taskId}, Details: ${JSON.stringify(taskPayload)}`);

            // Return the new task ID as string in 'text' and number in annotations
            return [{
                type: "text",
                text: taskId.toString(),
                annotations: { task_id_int: taskId } // Use snake_case for annotation key consistency?
            }];

        } catch (error: any) {
            logger.error(`❌ Error adding task: ${error.message}`, { stack: error.stack });
            // Re-throw the error to be caught by handleCallTool for proper JSON-RPC error response
            throw error;
        }
    }
};

// --- NEW Tool: Peek Task --- //
const peekTaskTool: ToolDefinition = {
    name: "_task_peek",
    description: "Retrieves the stored details for a specific task",
    inputSchema: {
        type: "object",
        properties: {
            id: { type: "integer", description: "ID of the task to peek" }
        },
        required: ["id"]
    },
    // Make 'id' non-optional
    async execute(args: { id: number }): Promise<TextContent[]> {
        logger.info(`👀 TASK PEEK CALLED with args: ${JSON.stringify(args)}`);
        try {
            const taskId = args.id;

            if (taskId === undefined || taskId === null) {
                throw new Error("Missing 'id' in arguments");
            }
            // Ensure it's treated as a number, specifically integer if needed, although TS handles number types
            if (typeof taskId !== 'number' || !Number.isInteger(taskId)) {
                 throw new Error("'id' must be an integer");
            }

            // Retrieve the task details from the map
            const taskDetails = tasks.get(taskId);

            if (taskDetails !== undefined) {
                logger.info(`✅ Task ${taskId} details found: ${JSON.stringify(taskDetails)}`);
                // Return the stored task details as JSON string in 'text' and raw dict in annotations.task_payload_json
                return [{
                    type: "text",
                    text: JSON.stringify(taskDetails),
                    annotations: { task_payload_json: taskDetails } // Add payload to annotations
                }];
            } else {
                logger.warn(`❓ Task ID ${taskId} not found.`);
                // Throw error to be handled by handleCallTool
                throw new Error(`Task ID ${taskId} not found`);
            }
        } catch (error: any) {
            logger.error(`❌ Error peeking task: ${error.message}`, { stack: error.stack });
            // Re-throw the error
            throw error;
        }
    }
};

// --- NEW Tool: Remove Task --- //
const removeTaskTool: ToolDefinition = {
    name: "_task_remove",
    description: "Removes a task",
    inputSchema: {
        type: "object",
        properties: {
            id: { type: "integer", description: "ID of the task to remove" }
        },
        required: ["id"]
    },
    // Make 'id' non-optional
    async execute(args: { id: number }): Promise<TextContent[]> {
        logger.info(`🗑️ TASK REMOVE CALLED with args: ${JSON.stringify(args)}`);
        try {
            const taskId = args.id;

            if (taskId === undefined || taskId === null) {
                throw new Error("Missing 'id' in arguments");
            }
            if (typeof taskId !== 'number' || !Number.isInteger(taskId)) {
                 throw new Error("'id' must be an integer");
            }

            // Attempt to delete the task from the map
            const deleted = tasks.delete(taskId);

            if (deleted) {
                logger.info(`✅ Task ${taskId} removed successfully.`);
                return [{ type: "text", text: `Task ${taskId} removed successfully.` }];
            } else {
                logger.warn(`❓ Task ID ${taskId} not found for removal.`);
                // Throw error as the task didn't exist
                throw new Error(`Task ID ${taskId} not found`);
            }
        } catch (error: any) {
            logger.error(`❌ Error removing task: ${error.message}`, { stack: error.stack });
            // Re-throw the error
            throw error;
        }
    }
};

// --- NEW Tool: Run Task --- //
const runTaskTool: ToolDefinition = {
    name: "_task_run",
    description: "Runs a dynamic Typescript function with the task data",
    inputSchema: {
        type: "object",
        properties: {
            id: { type: "integer", description: "ID of the task to run" }
        },
        required: ["id"]
    },
    // Make 'id' non-optional
    async execute(args: { id: number }): Promise<TextContent[]> {
        logger.info(`🏃 TASK RUN CALLED with args: ${JSON.stringify(args)}`);
        try {
            const taskId = args.id;

            if (taskId === undefined || taskId === null) {
                throw new Error("Missing 'id' in arguments");
            }
            if (typeof taskId !== 'number' || !Number.isInteger(taskId)) {
                 throw new Error("'id' must be an integer");
            }

            // Retrieve the task data (might contain payload and previous result)
            const taskData = tasks.get(taskId);

            if (!taskData) {
                throw new Error(`Task ID ${taskId} not found`);
            }

            // Extract payload - assume it might be nested if result is also stored
            const payload = taskData.payload || taskData; // Handle initial run vs subsequent
            if (!payload || typeof payload !== 'object'){
                 throw new Error(`Task ID ${taskId} does not contain a valid payload object.`);
            }

            const functionName = payload.functionName;
            const functionArgs = payload.arguments;

            if (typeof functionName !== 'string') {
                throw new Error(`Task ID ${taskId} payload missing 'functionName' string.`);
            }
             if (typeof functionArgs === 'undefined') { // Allow null/empty object args
                throw new Error(`Task ID ${taskId} payload missing 'arguments'.`);
            }

            // Find the dynamic function tool definition
            const tool = toolRegistry.get(functionName);
            if (!tool || !tool.execute) { // Check if it's an executable tool
                 throw new Error(`Dynamic function tool '${functionName}' not found or not executable.`);
            }
            // Optional: Add check to ensure it's actually a *dynamic* function if needed

            logger.info(`🚀 Executing dynamic function '${functionName}' for task ${taskId} with args: ${JSON.stringify(functionArgs)}`);

            // Execute the dynamic function
            const result = await tool.execute(functionArgs);

            logger.info(`✅ Dynamic function '${functionName}' for task ${taskId} completed.`);

            // Store the result back with the task data
            // Preserve original payload, add/update result
            tasks.set(taskId, { payload: payload, result: result });

            return [{ type: "text", text: `Task ${taskId} executed successfully, result stored.` }];

        } catch (error: any) {
            logger.error(`❌ Error running task ${args.id}: ${error.message}`, { stack: error.stack });
            // Re-throw the error
            throw error;
        }
    }
};

// --- Existing Stubs (Keep for now, implement later) ---
// const taskAddStub = createStubTool("_task_add", "Adds a task.", { payload: { type: "object" } }, ["payload"]); // Replaced by addTaskTool
// const taskRunStub = createStubTool("_task_run", "Runs a dynamic Typescript function with the task data", { id: { type: "integer" } }, ["id"]); // Replaced by runTaskTool
// const taskRemoveStub = createStubTool("_task_remove", "Removes a task.", { id: { type: "integer" } }, ["id"]); // Replaced by removeTaskTool
// const taskPeekStub = createStubTool("_task_peek", "Peeks a task.", { id: { type: "integer" } }, ["id"]); // Replaced by peekTaskTool

// Register built-in tools
toolRegistry.set(registerFunctionTool.name, registerFunctionTool);
toolRegistry.set(getFunctionCodeTool.name, getFunctionCodeTool);
toolRegistry.set(removeFunctionTool.name, removeFunctionTool);
toolRegistry.set(addFunctionTool.name, addFunctionTool);

// Register Task tools
toolRegistry.set(addTaskTool.name, addTaskTool); // Register the real implementation
toolRegistry.set(runTaskTool.name, runTaskTool); // Register the real implementation
toolRegistry.set(removeTaskTool.name, removeTaskTool); // Register the real implementation
toolRegistry.set(peekTaskTool.name, peekTaskTool); // Register the real implementation

// --- MCP Request Handlers ---
const handleListTools = (ws: WebSocket, id: string | number | null, _params: any): void => {
    const toolsForClient = prepareToolsListPayload(); // Use the helper
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
            // IMPORTANT: We use "contents" (plural) key to match format between Python and Node servers
            // Both MCP SDK implementations support either "content" or "contents" but we need to be consistent
            sendResponse(ws, id, { contents: result });

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
                logger.warn(`PID file (${PID_FILE}) contains invalid content: "${pidString}". Ignoring and removing.`);
                try { unlinkSync(PID_FILE); } catch (unlinkErr: any) { logger.error(`Failed to remove invalid PID file: ${unlinkErr.message}`); }
            }
        } catch (readError: any) {
            logger.error(`❌ Error reading PID file (${PID_FILE}): ${readError.message}. Exiting.`);
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
checkAndHandleExistingProcess(logger, PID_FILE); // <-- FIXED CALL

// If the check didn't exit, start the server
startServer().catch(error => {
    logger.error(`🚨 Failed to start server: ${error.message}`, { stack: error.stack });
    process.exit(1);
});
