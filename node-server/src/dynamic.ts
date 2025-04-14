import * as ts from 'typescript'; // For TypeScript compiler API
import * as path from 'path';
import * as fsPromises from 'fs/promises';
import { existsSync, readFileSync, statSync } from 'fs';
import * as winston from 'winston';
import { ToolDefinition, DynamicFunctionModule } from './types';

// These will be set when initialized
let logger: winston.Logger;
let toolRegistry: Map<string, ToolDefinition>;
let dynamicFunctionFiles: Map<string, string>;
let SOURCE_FUNCTIONS_DIR: string;

// Initialize the module with necessary dependencies
export function initializeDynamic(
    loggerInstance: winston.Logger,
    toolRegistryMap: Map<string, ToolDefinition>,
    dynamicFunctionFilesMap: Map<string, string>,
    sourceFunctionsDir: string
): void {
    logger = loggerInstance;
    toolRegistry = toolRegistryMap;
    dynamicFunctionFiles = dynamicFunctionFilesMap;
    SOURCE_FUNCTIONS_DIR = sourceFunctionsDir;
    
    logger.debug('Dynamic functions module initialized');
}

// --- Helper Function for Formatting Tool List ---
export const prepareToolsListPayload = (): any[] => {
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
export const loadAndRegisterDynamicFunction = async (compiledFilePath: string): Promise<void> => {
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
        const stats = await fsPromises.stat(compiledFilePath);

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
export const extractSchemaFromTypeScript = (sourceCode: string, functionName: string): any => {
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
export const extractParamsFromFunction = (node: ts.FunctionLikeDeclaration,
                                 properties: Record<string, any>,
                                 required: string[]): void => {
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
export async function inspectTypeScriptFunction(code: string, exportedFunctionName: string): Promise<{ description: string, inputSchema: any }> {
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

