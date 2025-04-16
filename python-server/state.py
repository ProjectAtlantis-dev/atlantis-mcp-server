#!/usr/bin/env python3
import logging
import os

# Configure logging - focus on our application logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
# Set our app logger to DEBUG
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.DEBUG)

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
