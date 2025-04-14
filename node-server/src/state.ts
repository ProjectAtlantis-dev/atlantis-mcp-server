/**
 * Global state file for sharing variables between modules
 * This centralizes our global state management
 */
import winston from 'winston'; 
import { ToolDefinition } from './types';

// --- Private state variables ---
let _logger: winston.Logger;
const _toolRegistry = new Map<string, ToolDefinition>();
const _dynamicFunctionFiles = new Map<string, string>();
let _cloudSocket: any = null;
let _cloudReconnectTimer: NodeJS.Timeout | null = null; 
let _cloudConnectionAttempts: number = 0;
let _COMPILED_FUNCTIONS_DIR: string;
let _SOURCE_FUNCTIONS_DIR: string;

// --- Getters ---
export const getLogger = (): winston.Logger => _logger;
export const getToolRegistry = (): Map<string, ToolDefinition> => _toolRegistry;
export const getDynamicFunctionFiles = (): Map<string, string> => _dynamicFunctionFiles;
export const getCloudSocket = (): any => _cloudSocket;
export const getCloudReconnectTimer = (): NodeJS.Timeout | null => _cloudReconnectTimer;
export const getCloudConnectionAttempts = (): number => _cloudConnectionAttempts;
export const getCompiledFunctionsDir = (): string => _COMPILED_FUNCTIONS_DIR;
export const getSourceFunctionsDir = (): string => _SOURCE_FUNCTIONS_DIR;

// --- Setters ---
export const setCloudSocket = (socket: any): void => { _cloudSocket = socket; };
export const setCloudReconnectTimer = (timer: NodeJS.Timeout | null): void => { _cloudReconnectTimer = timer; };
export const setCloudConnectionAttempts = (attempts: number): void => { _cloudConnectionAttempts = attempts; };

// --- Helper functions ---
export const incrementCloudConnectionAttempts = (): number => { 
    _cloudConnectionAttempts++; 
    return _cloudConnectionAttempts;
};

export const resetCloudConnectionAttempts = (): void => { 
    _cloudConnectionAttempts = 0; 
};

/**
 * Initialize the shared state
 */
export function initializeState(
    loggerInstance: winston.Logger,
    compiledFunctionsDir: string,
    sourceFunctionsDir: string
): void {
    _logger = loggerInstance;
    _COMPILED_FUNCTIONS_DIR = compiledFunctionsDir;
    _SOURCE_FUNCTIONS_DIR = sourceFunctionsDir;
    
    _logger.debug('🌟 Global state initialized');
}
