import { existsSync, readFileSync, unlinkSync } from 'fs';
import process from 'process'; // Explicit import for process
import winston from 'winston'; // Import winston type for the logger parameter

// --- Function to Check Existing Process --- //
export function checkAndHandleExistingProcess(logger: winston.Logger, PID_FILE: string): void {
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
