#!/usr/bin/env python3
import logging
import os
import sys

# ANSI escape codes for colors
GREY = "\x1b[90m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
BOLD_RED = "\x1b[31;1m"
RESET = "\x1b[0m"
GREEN = "\x1b[32m" # Added Green for INFO
BOLD = "\x1b[1m"   # Added Bold
CYAN = "\x1b[36m"   # Added Cyan
BRIGHT_WHITE = "\x1b[97m" # Added Bright White

# Custom Formatter
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.INFO: GREEN + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.WARNING: YELLOW + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.ERROR: RED + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, logging.BASIC_FORMAT)
        # Use a specific date format
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# --- REMOVED basicConfig ---

# Get our app logger
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.DEBUG)

# --- ADDED Handler setup ---
# Create console handler
ch = logging.StreamHandler(sys.stdout) # Use stdout
ch.setLevel(logging.DEBUG) # Process all messages from logger

# Set the custom formatter
ch.setFormatter(ColoredFormatter())

# Add handler to the logger
logger.addHandler(ch)

# Prevent logging from propagating to the root logger
# (important if basicConfig was ever called or might be by libraries)
logger.propagate = False
# --- End Handler setup ---

# Directory to store dynamic function files
FUNCTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_functions")

# Create functions directory if it doesn't exist
os.makedirs(FUNCTIONS_DIR, exist_ok=True)

# Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces by default
PORT = 8000

# Cloud server configuration
CLOUD_SERVER_HOST = "localhost"
CLOUD_SERVER_PORT = 3010
CLOUD_SERVER_URL = f"http://{CLOUD_SERVER_HOST}:{CLOUD_SERVER_PORT}"
CLOUD_SERVICE_NAMESPACE = "/service"  # Socket.IO namespace for service-to-service communication
CLOUD_CONNECTION_RETRY_SECONDS = 5  # Initial delay in seconds
CLOUD_CONNECTION_MAX_RETRIES = 10  # Maximum number of retries before giving up (None for infinite)
CLOUD_CONNECTION_MAX_BACKOFF_SECONDS = 60  # Maximum delay for exponential backoff

# Flags to track server state
is_shutting_down = False
cloud_connection_active = False

# Dictionary to store task information
tasks = {}
next_task_id = 1
