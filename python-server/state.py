#!/usr/bin/env python3
import logging
import os
import sys
from ColoredFormatter import ColoredFormatter

# --- REMOVED basicConfig ---

# Determine log level from environment variable if set, otherwise default to INFO
default_level = os.environ.get('LOG_LEVEL', 'INFO')
log_level = getattr(logging, default_level, logging.INFO)

# Get our app logger
logger = logging.getLogger("mcp_server")
logger.setLevel(log_level)

# --- ADDED Handler setup ---
# Create console handler
ch = logging.StreamHandler(sys.stdout) # Use stdout
ch.setLevel(log_level) # Process all messages from logger

# Set the custom formatter
ch.setFormatter(ColoredFormatter())

# Add handler to the logger
logger.addHandler(ch)

# Configure the root logger to also use this handler and level
root_logger = logging.getLogger()
# Set root logger level (e.g., INFO to see messages from message_db.py)
root_logger.setLevel(logging.INFO)

# Add the same handler to the root logger if it doesn't have any
# This ensures messages from 'logging.info()' etc. in other modules are also seen
if not root_logger.hasHandlers():
    root_logger.addHandler(ch)

# Prevent logging from propagating to the root logger
# (important if basicConfig was ever called or might be by libraries)
logger.propagate = False
# --- End Handler setup ---

def update_log_level(level_name):
    """Update the logging level of both loggers and handlers.
    
    Args:
        level_name: String name of logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    level = getattr(logging, level_name)  # Convert string to logging level constant
    logger.setLevel(level)
    ch.setLevel(level)
    
    # Also update root logger if it's using our handler
    if level_name != "DEBUG":  # Keep root at INFO or higher
        root_logger.setLevel(level)
    
    logger.info(f"Log level updated to {level_name}")

# Directory to store dynamic function files
FUNCTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_functions")

# Create functions directory if it doesn't exist
os.makedirs(FUNCTIONS_DIR, exist_ok=True)

# Directory to store dynamic server configs
SERVERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_servers")
# Create servers directory if it doesn't exist
os.makedirs(SERVERS_DIR, exist_ok=True)

# Server configuration
HOST = "127.0.0.1"  # Listen on localhost only for security
PORT = 8000

SERVER_REQUEST_TIMEOUT = 3600.0 # Seconds to wait for proxied server requests and awaitable client commands (1 hour for long-running cloud jobs)

# Flags to track server state
is_shutting_down = False

