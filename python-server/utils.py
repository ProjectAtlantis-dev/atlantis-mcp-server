#!/usr/bin/env python3
import os
import psutil
import logging

# Set up logger
logger = logging.getLogger("mcp_server")

# Path for the PID file
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.pid")

def check_server_running():
    """Check if a server is already running by examining the PID file.
    
    Returns:
        int or None: The PID of the running server, or None if no server is running.
    """
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())

            # Check if the process with this PID exists
            if psutil.pid_exists(pid):
                # Get process name to confirm it's our server
                process = psutil.Process(pid)
                if "python" in process.name().lower() and any("server.py" in cmd.lower() for cmd in process.cmdline()):
                    return pid

            # If we get here, the PID exists but it's not our server process
            logger.warning(f"🧹 Removing stale PID file from previous server instance")
            os.remove(PID_FILE)
            return None
        except (ValueError, ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"🧹 Removing stale PID file: {str(e)}")
            os.remove(PID_FILE)
            return None
    return None

def create_pid_file():
    """Create a PID file with the current process ID.
    
    Returns:
        bool: True if the file was created successfully, False otherwise.
    """
    pid = os.getpid()
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(pid))
        logger.info(f"📝 Created PID file with server process ID: {pid}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create PID file: {str(e)}")
        return False

def remove_pid_file():
    """Remove the PID file if it exists."""
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
            logger.info(f"🧹 Removed PID file")
    except Exception as e:
        logger.error(f"❌ Failed to remove PID file: {str(e)}")
