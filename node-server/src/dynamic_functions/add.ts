// src/dynamic_functions/add.ts
import { InputSchema } from '../types'; // Import InputSchema

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
    } as InputSchema // Cast to ensure type correctness
};

// Define the actual function logic
// Can return simple value or the structured TextContent[]
export const handler = async (args: { a: number, b: number }): Promise<string> => {
    // Input args are already validated basic presence by handleCallTool (if required specified)
    // TODO: Add more robust validation within the handler if needed
    const sum = args.a + args.b;
    return `The sum of ${args.a} and ${args.b} is: ${sum}`;
};
