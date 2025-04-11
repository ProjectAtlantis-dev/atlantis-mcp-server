// src/dynamic_functions/add.ts
import { InputSchema } from '../types';

// Define the metadata for the MCP tool registry
export const metadata = {
    description: "Adds two numbers together.",
    inputSchema: {
        type: "object",
        properties: {
            a: { type: "number", description: "The first number." },
            b: { type: "number", description: "The second number." }
        },
        required: ["a", "b"]
    } as InputSchema
};

// Define the actual function logic, returning a raw number
export const handler = async (a: number, b: number): Promise<number> => { 
    // TODO: Add more robust validation within the handler if needed?
    const sum = a + b;
    return sum; 
};
