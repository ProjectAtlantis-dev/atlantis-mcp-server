#!/usr/bin/env node

const WebSocketClient = require('websocket').client;
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const fs = require('fs');
const path = require('path');

// Default configuration values
const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 8000;
const DEFAULT_PATH = '/mcp';

// Helper function to output JSON messages to stdout
function jsonLog(message, level = 'info') {
  const jsonOutput = {
    type: 'text',
    text: message,
    metadata: {
      level: level,
      timestamp: new Date().toISOString()
    }
  };
  process.stdout.write(JSON.stringify(jsonOutput) + '\n');
}

// Override console methods to use our JSON logger
console.log = (message) => jsonLog(message, 'info');
console.warn = (message) => jsonLog(message, 'warning');
console.error = (message) => jsonLog(message, 'error');

// Parse command line arguments
const argv = yargs(hideBin(process.argv))
  .option('host', {
    alias: 'h',
    type: 'string',
    description: 'Host of the Python MCP server',
    default: DEFAULT_HOST
  })
  .option('port', {
    alias: 'p',
    type: 'number',
    description: 'Port of the Python MCP server',
    default: DEFAULT_PORT
  })
  .option('path', {
    type: 'string',
    description: 'WebSocket path',
    default: DEFAULT_PATH
  })
  .help()
  .argv;

// Check if server is running by checking PID file
function checkServerRunning() {
  // Only perform PID check when connecting to localhost
  if (argv.host !== '127.0.0.1') {
    console.log(`PID file checking skipped (not connecting to localhost)`);
    return true;
  }

  // Determine PID file location
  const pidFilePath = path.join(__dirname, '..', 'mcp_server.pid');

  if (fs.existsSync(pidFilePath)) {
    try {
      const pid = parseInt(fs.readFileSync(pidFilePath, 'utf8').trim());
      console.log(`Found PID file with server process ID: ${pid}`);
      return true;
    } catch (error) {
      console.warn(`Found PID file but couldn't read it: ${error.message}`);
      return false;
    }
  } else {
    console.warn(`No PID file found at ${pidFilePath}`);
    console.warn(`Make sure the Python server is running before starting this bridge!`);
    return true;
  }
}

// Define main MCP bridge function
async function startMcpBridge() {
  // Check if server appears to be running first
  if (!checkServerRunning()) {
    console.warn(`Server may not be running. Will attempt to connect anyway...`);
  }

  // Construct server URL
  const serverUrl = `ws://${argv.host}:${argv.port}${argv.path}`;

  console.log(`MCP Python Bridge starting...`);
  console.log(`Connecting to Python MCP server at: ${serverUrl}`);

  // Set up standard input/output for passthrough
  process.stdin.setEncoding('utf8');

  // Create WebSocket client instance
  const client = new WebSocketClient();

  // Handle connection failures
  client.on('connectFailed', (error) => {
    console.error(`Connection error: ${error.toString()}`);
    process.exit(1);
  });

  // Handle successful connections
  client.on('connect', (connection) => {
    console.log(`Connected to Python MCP server!`);
    console.log(`Bridge ready! Passing messages between process and Python server...`);

    // Handle messages from the WebSocket
    connection.on('message', (message) => {
      try {
        if (message.type === 'utf8') {
          const wsData = message.utf8Data;

          // Attempt to parse as JSON (in case server sends JSON objects)
          try {
            // Parse and re-stringify to ensure proper JSON formatting
            const jsonData = JSON.parse(wsData);

            // Write the properly formatted JSON to stdout
            process.stdout.write(JSON.stringify(jsonData) + '\n');
          } catch (parseError) {
            // If it's not valid JSON, wrap it in a JSON text content object
            const jsonWrapper = {
              type: "text",
              text: wsData
            };
            process.stdout.write(JSON.stringify(jsonWrapper) + '\n');
          }
        }
      } catch (error) {
        console.error(`Error processing server message: ${error.message}`);
      }
    });

    // Handle connection closure
    connection.on('close', () => {
      console.log(`Connection to Python MCP server closed`);
      process.exit(0);
    });

    // Handle connection errors
    connection.on('error', (error) => {
      console.error(`Connection error: ${error.toString()}`);
    });

    // Buffer for collecting incoming data chunks
    let inputBuffer = '';

    // Set up stdin to forward to WebSocket
    process.stdin.on('data', (data) => {
      try {
        // Add the new data to our buffer
        inputBuffer += data.toString();

        // Process complete JSON objects
        let newlineIndex;
        while ((newlineIndex = inputBuffer.indexOf('\n')) !== -1) {
          // Extract a complete line
          const line = inputBuffer.slice(0, newlineIndex).trim();
          inputBuffer = inputBuffer.slice(newlineIndex + 1);

          if (!line) continue; // Skip empty lines

          try {
            // Parse the JSON input
            const jsonInput = JSON.parse(line);

            // Extract the actual text content if it's wrapped in a standard format
            let textToSend;
            if (jsonInput.type === 'text' && jsonInput.text) {
              textToSend = jsonInput.text;
            } else {
              // Otherwise send the stringified JSON
              textToSend = JSON.stringify(jsonInput);
            }

            if (connection.connected) {
              connection.sendUTF(textToSend);
            }
          } catch (jsonError) {
            // If it's not valid JSON, send it as is
            if (connection.connected && line) {
              connection.sendUTF(line);
            }
          }
        }
      } catch (error) {
        console.error(`Error forwarding input to server: ${error.message}`);
      }
    });
  });

  // Handle process termination
  process.on('SIGINT', () => {
    console.log(`Received SIGINT, shutting down MCP bridge...`);
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    console.log(`Received SIGTERM, shutting down MCP bridge...`);
    process.exit(0);
  });

  // Connect to WebSocket server
  client.connect(serverUrl, 'mcp');
}

// Start the bridge
startMcpBridge().catch(error => {
  console.error(`Fatal error: ${error.message}`);
  process.exit(1);
});
