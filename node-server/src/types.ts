// src/types.ts

// JSON-RPC Interfaces
export interface JsonRpcRequest {
    jsonrpc: '2.0';
    method: string;
    params?: any;
    id: string | number | null;
}

export interface JsonRpcResponse {
    jsonrpc: '2.0';
    result?: any;
    error?: JsonRpcError;
    id: string | number | null;
}

export interface JsonRpcError {
    code: number;
    message: string;
    data?: any;
}

// MCP Content Types
export interface TextContent {
    type: 'text';
    text: string;
}
// Add other content types if needed

// Tool Definition Interfaces
export interface ToolParameterProperty {
    type: string;
    description?: string;
    // Add other JSON Schema properties if needed (e.g., enum, format, default)
}

export interface ToolParameters {
    type: 'object';
    properties: { [key: string]: ToolParameterProperty };
    required?: string[];
}

export interface ToolMetadata {
    description: string;
    parameters?: ToolParameters;
}

export interface ToolDefinition extends ToolMetadata {
    name: string;
    // --- Internal fields ---
    execute?: (args: any) => Promise<TextContent[]>; // For built-ins/stubs
    _function?: Function; // For dynamically loaded functions
    _filePath?: string; // Path to the loaded JS file
}

// Dynamic Module structure (for loaded JS files)
export interface DynamicFunctionModule {
    handler: Function; // Assuming the main function is exported as 'handler'
    metadata: ToolMetadata; // Assuming metadata is exported as 'metadata'
}
