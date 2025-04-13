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
import * as ts from 'typescript'; // Added for Compiler API

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
    async execute(args: { code?: string }): Promise<TextContent[]> {
        const code = args.code;

        if (!code) {
            throw new Error("Missing required argument: code");
        }

        // --- Extract function name from code --- PRE-SAVE STEP
        let functionName: string | null = null;
        try {
            const sourceFile = ts.createSourceFile(
                'temp.ts', // Temporary filename for parsing
                code,
                ts.ScriptTarget.Latest,
                true // Set parent pointers
            );

            ts.forEachChild(sourceFile, node => {
                if (functionName) return; // Stop searching once found

                if (ts.isVariableStatement(node) && node.modifiers?.some(mod => mod.kind === ts.SyntaxKind.ExportKeyword)) {
                    node.declarationList.declarations.forEach(declaration => {
                        if (declaration.name.getText(sourceFile) === 'toolDefinition' && declaration.initializer && ts.isObjectLiteralExpression(declaration.initializer)) {
                            declaration.initializer.properties.forEach(prop => {
                                if (ts.isPropertyAssignment(prop) && prop.name.getText(sourceFile) === 'name' && prop.initializer && ts.isStringLiteral(prop.initializer)) {
                                    functionName = prop.initializer.text; // Extract the string value
                                }
                            });
                        }
                    });
                }
            });

            if (!functionName) {
                throw new Error("Could not find an exported 'toolDefinition' constant with a string 'name' property in the provided code.");
            }
             logger.info(`Extracted function name from code: ${functionName}`);

        } catch (parseError: any) {
            logger.error(`Failed to parse code or extract function name: ${parseError.message}`);
            throw new Error(`Failed to parse code: ${parseError.message}`);
        }
        // --- End extraction ---

        // Basic validation for extracted function name
        if (!/^[a-zA-Z0-9_]+$/.test(functionName)) {
            throw new Error(`Invalid extracted function_name '${functionName}': only alphanumeric characters and underscores allowed.`);
        }

        const sourceFilePath = path.join(SOURCE_FUNCTIONS_DIR, `${functionName}.ts`);
        const compiledFilePath = path.join(COMPILED_FUNCTIONS_DIR, `${functionName}.js`);

        logger.info(`Attempting to register function '${functionName}' (name derived from code)...`);

        // 1. Save the TypeScript code (now that we know the name)
        try {
            // Ensure the source directory exists
            await fs.mkdir(SOURCE_FUNCTIONS_DIR, { recursive: true });
            await fs.writeFile(sourceFilePath, code, 'utf8');
            logger.info(`Saved TypeScript source to: ${sourceFilePath}`);
        } catch (writeError: any) {
            logger.error(`Failed to write source file ${sourceFilePath}: ${writeError.message}`);
            throw new Error(`Failed to save function source: ${writeError.message}`);
        }

        // 2. Attempt to compile the TypeScript file using the Compiler API
        try {
            // Ensure compiled directory exists
            await fs.mkdir(COMPILED_FUNCTIONS_DIR, { recursive: true });

            const compilerOptions: ts.CompilerOptions = {
                outDir: COMPILED_FUNCTIONS_DIR,
                target: ts.ScriptTarget.ES2016,
                module: ts.ModuleKind.CommonJS,
                esModuleInterop: true,
                skipLibCheck: true,
                resolveJsonModule: true,
                // sourceMap: true, // Optional: generate source maps
                declaration: false, // Optional: don't generate .d.ts files
            };

            logger.info(`Compiling ${sourceFilePath} using TypeScript API...`);
            const program = ts.createProgram([sourceFilePath], compilerOptions);
            const emitResult = program.emit();

            const allDiagnostics = ts
                .getPreEmitDiagnostics(program)
                .concat(emitResult.diagnostics);

            let hasError = false;
            const diagnosticMessages: string[] = [];

            allDiagnostics.forEach(diagnostic => {
                if (diagnostic.category === ts.DiagnosticCategory.Error) {
                    hasError = true;
                }
                if (diagnostic.file) {
                    const { line, character } = ts.getLineAndCharacterOfPosition(diagnostic.file, diagnostic.start!);
                    const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
                    diagnosticMessages.push(`${path.basename(diagnostic.file.fileName)} (${line + 1},${character + 1}): ${message}`);
                } else {
                    diagnosticMessages.push(ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
                }
            });

            if (hasError || emitResult.emitSkipped) {
                const errorOutput = diagnosticMessages.join('\n');
                logger.error(`Compilation failed for ${functionName}. Diagnostics:\n${errorOutput}`);
                // Attempt to clean up the bad .ts file
                try { await fs.unlink(sourceFilePath); } catch { /* Ignore cleanup error */ }
                throw new Error(`Compilation failed:\n${errorOutput}`);
            }

            logger.info(`Successfully compiled ${functionName} using TypeScript API to: ${compiledFilePath}`);

            // 3. Construct and Register the ToolDefinition WRAPPER
            try {
                 // Ensure require cache is clear before loading the new module
                const absolutePath = require.resolve(compiledFilePath);
                delete require.cache[absolutePath];
                logger.debug(`Cleared require cache for compiled file: ${absolutePath}`)

                const dynamicModule = require(absolutePath);

                // --- Find the single exported function --- START NEW LOGIC ---
                let userFunctionName: string | null = null;
                let userFunction: Function | null = null;
                const exports = Object.keys(dynamicModule);
                // Filter out non-function exports and potentially internal properties like __esModule
                const potentialFunctions = exports.filter(key => typeof dynamicModule[key] === 'function' && key !== '__esModule');

                if (potentialFunctions.length === 0) {
                    // Check for default export if no named functions found
                    if (dynamicModule.default && typeof dynamicModule.default === 'function') {
                        logger.info(`Found default export function in module ${functionName}.js`);
                        userFunction = dynamicModule.default;
                        userFunctionName = 'default'; // Representing the default export
                    } else {
                        throw new Error(`Compiled module ${functionName}.js does not export any functions (checked named and default).`);
                    }
                } else if (potentialFunctions.length > 1) {
                     // Decision: Require exactly one named export OR a default export if no named ones exist.
                     // If both exist, or multiple named exports exist, it's ambiguous.
                     if (dynamicModule.default && typeof dynamicModule.default === 'function') {
                           throw new Error(`Compiled module ${functionName}.js exports multiple named functions (${potentialFunctions.join(', ')}) AND a default export. Please export only ONE function (either named or default).`);
                     } else {
                           throw new Error(`Compiled module ${functionName}.js exports multiple named functions (${potentialFunctions.join(', ')}). Please export only one function.`);
                     }
                } else {
                    // Exactly one named function found
                    userFunctionName = potentialFunctions[0];
                    userFunction = dynamicModule[userFunctionName];
                    logger.info(`Found single exported function '${userFunctionName}' in module ${functionName}.js`);
                }
                // --- End find function ---

                if (!userFunction) { // Should be unreachable if logic above is sound
                     throw new Error(`Could not locate the function to execute within ${functionName}.js`);
                }

                // Construct the ToolDefinition dynamically
                const constructedTool: ToolDefinition = {
                    name: functionName,
                    description: dynamicModule.metadata.description,
                    inputSchema: dynamicModule.metadata.inputSchema, // Use the schema provided by the user
                    async execute(toolArgs: any): Promise<TextContent[]> {
                        // Use the actual exported function name found, or 'default'
                        const actualUserFuncName = userFunctionName || 'default';
                        logger.info(`Executing dynamic tool '${functionName}' (wraps '${actualUserFuncName}') with args: ${JSON.stringify(toolArgs)}`);

                        // --- Prepare arguments based on schema order --- START NEW LOGIC ---
                        let orderedArgs: any[] = [];
                        // Ensure toolArgs is an object, default to empty if null/undefined
                        const currentToolArgs = (toolArgs && typeof toolArgs === 'object') ? toolArgs : {};

                        if (dynamicModule.metadata.inputSchema && dynamicModule.metadata.inputSchema.properties && typeof dynamicModule.metadata.inputSchema.properties === 'object') {
                            // Get argument names *in the order they appear in the schema's properties object*
                            const argNames = Object.keys(dynamicModule.metadata.inputSchema.properties);
                            orderedArgs = argNames.map(argName => {
                                // Get value from the arguments passed to the tool's execute function
                                if (currentToolArgs.hasOwnProperty(argName)) {
                                    return currentToolArgs[argName];
                                } else {
                                    // Argument not provided. Check if it was required in the schema.
                                    const isRequired = dynamicModule.metadata.inputSchema.required && dynamicModule.metadata.inputSchema.required.includes(argName);
                                    if (isRequired) {
                                        // This case should ideally be caught by MCP framework's validation before calling execute,
                                        // but throw an error here just in case.
                                        logger.error(`Required argument '${argName}' missing for tool '${functionName}'.`);
                                        throw new Error(`Required argument '${argName}' is missing.`);
                                    } else {
                                        // Argument is optional and not provided, pass undefined to the user function
                                        logger.debug(`Optional argument '${argName}' not provided for tool '${functionName}', using undefined.`);
                                        return undefined;
                                    }
                                }
                            });
                             logger.debug(`Prepared ordered args for '${actualUserFuncName}': ${JSON.stringify(orderedArgs)}`);
                        } else {
                             logger.info(`No inputSchema.properties found for '${functionName}'. Calling user function '${actualUserFuncName}' with no arguments.`);
                             // Call with no arguments if schema has no properties
                             orderedArgs = [];
                        }
                        // --- End prepare arguments ---

                        try {
                            // Call the USER'S pure function with individual, ordered arguments
                            const result = await userFunction(...orderedArgs);

                            // Basic result formatting (adapt as needed)
                            let textResult: string;
                            if (typeof result === 'string') {
                                textResult = result;
                            } else if (result === undefined || result === null) {
                                 textResult = `Function '${functionName}' executed successfully with no return value.`;
                            } else {
                                // Attempt to stringify other types
                                try {
                                    textResult = JSON.stringify(result, null, 2);
                                } catch (stringifyError: any) {
                                     logger.error(`Failed to stringify result for '${functionName}': ${stringifyError.message}`);
                                     textResult = `[Unstringifiable Result: ${stringifyError.message}]`;
                                }
                            }
                            logger.info(`Dynamic tool '${functionName}' execution successful.`);
                            return [{ type: "text", text: textResult }];

                        } catch (runError: any) {
                            logger.error(`Error during dynamic tool '${functionName}' execution (calling '${actualUserFuncName}'): ${runError.message}`);
                            const errorMessage = runError.stack ? runError.stack : runError.message;
                            throw new Error(`Execution failed in '${functionName}': ${errorMessage}`);
                        }
                    }
                };

                // Register the CONSTRUCTED tool
                if (toolRegistry.has(functionName)) {
                     logger.warn(`Overwriting existing tool in registry: ${functionName}`);
                }
                toolRegistry.set(functionName, constructedTool);
                dynamicFunctionFiles.set(functionName, compiledFilePath); // Track the file
                logger.info(`Successfully constructed and registered tool wrapper: ${functionName}`);
                return [{ type: "text", text: `Function '${functionName}' registered and compiled successfully.` }];

            } catch (loadOrWrapError: any) {
                 logger.error(`Compilation succeeded but failed to load/wrap function ${functionName}: ${loadOrWrapError.message}`);
                 // Clean up both files on load/wrap error
                 try { await fs.rm(compiledFilePath); } catch { /* Ignore */ }
                 try { await fs.rm(sourceFilePath); } catch { /* Ignore */ }
                 throw new Error(`Compilation succeeded but failed during registration: ${loadOrWrapError.message}`);
            }

        } catch (compileOrRegisterError: any) {
            // This catches errors from file operations, API usage, or the re-thrown errors above
            logger.error(`Registration process failed for ${functionName}: ${compileOrRegisterError.message}`);
            // Attempt to clean up the .ts file if it still exists and wasn't cleaned up above
            try {
                if (existsSync(sourceFilePath)) { await fs.rm(sourceFilePath); }
            } catch { /* Ignore */ }
            // Re-throw the specific error
            throw compileOrRegisterError; // Propagate the original error object
        }
    }
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
    async execute(args: { name?: string }): Promise<TextContent[]> { // Changed param name in type
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
    async execute(args: { name?: string }): Promise<TextContent[]> {
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
    async execute(args: { name?: string }): Promise<TextContent[]> {
        const name = args.name;

        if (!name) {
            throw new Error("Missing required argument: name");
        }

        logger.info(`➕ ADDING NEW PLACEHOLDER FUNCTION: ${name}`);

        // Define the placeholder content using the provided name
        const placeholderCode = `// Placeholder function created by _function_add

// This definition is required by _function_register
export const toolDefinition = {
    name: "${name}", // Using the provided name
    description: "A newly added placeholder function. Implement your logic here.",
    inputSchema: {
        type: "object",
        properties: {} // No input arguments
    }
};

// Metadata is also required for the description
export const metadata = {
    description: "A newly added placeholder function. Implement your logic here."
};

// The actual function that gets executed
export function handler(): string {
    console.log(\`Executing placeholder function '${name}'\`);
    // Add your logic here!
    return "Placeholder function executed successfully.";
}`;
        const placeholderDescription = "A newly added placeholder function. Implement your logic here.";
        const placeholderSchema = { type: "object", properties: {} }; // No input arguments

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
    async execute(args: { payload?: any }): Promise<TextContent[]> {
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
    async execute(args: { id?: number }): Promise<TextContent[]> {
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
    async execute(args: { id?: number }): Promise<TextContent[]> {
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
    async execute(args: { id?: number }): Promise<TextContent[]> {
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
    const authPayload = { apiKey, email, serviceName, hostname: os.hostname() };
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
                                // IMPORTANT: We use "contents" (plural) key to match format between Python and Node servers
                                // Both MCP SDK implementations support either "content" or "contents" but we need to be consistent
                                result: { contents: resultContents },
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
