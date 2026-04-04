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

function writeStdioMessage(payload) {
  const body = typeof payload === 'string' ? payload : JSON.stringify(payload);
  process.stdout.write(body + '\n');
}

function trimForLog(value, maxLength = 240) {
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength)}...`;
}

function describeJsonRpcMessage(payload) {
  if (!payload || typeof payload !== 'object') {
    return `non-object payload (${typeof payload})`;
  }

  if (payload.method) {
    return `request method=${payload.method} id=${payload.id ?? 'notification'}`;
  }

  if (Object.prototype.hasOwnProperty.call(payload, 'result')) {
    return `response id=${payload.id ?? 'unknown'} resultKeys=${Object.keys(payload.result || {}).join(',') || 'none'}`;
  }

  if (payload.error) {
    const code = payload.error.code ?? 'unknown';
    const message = payload.error.message ?? trimForLog(payload.error);
    return `error id=${payload.id ?? 'unknown'} code=${code} message=${message}`;
  }

  return `json-rpc payload keys=${Object.keys(payload).join(',')}`;
}

function isHandshakeMessage(payload) {
  if (!payload || typeof payload !== 'object') {
    return false;
  }

  return payload.method === 'initialize'
    || payload.method === 'notifications/initialized'
    || Object.prototype.hasOwnProperty.call(payload, 'result')
    || Object.prototype.hasOwnProperty.call(payload, 'error');
}

function tryConsumeStdioFrame(buffer) {
  const headerEnd = buffer.indexOf('\r\n\r\n');
  if (headerEnd === -1) {
    return null;
  }

  const headerBlock = buffer.slice(0, headerEnd);
  const contentLengthMatch = headerBlock.match(/(?:^|\r\n)Content-Length:\s*(\d+)(?:\r\n|$)/i);
  if (!contentLengthMatch) {
    return null;
  }

  const contentLength = Number.parseInt(contentLengthMatch[1], 10);
  const bodyStart = headerEnd + 4;
  if (buffer.length < bodyStart + contentLength) {
    return null;
  }

  const body = buffer.slice(bodyStart, bodyStart + contentLength);
  const rest = buffer.slice(bodyStart + contentLength);
  return { body, rest };
}

function tryConsumeLegacyLine(buffer) {
  const newlineIndex = buffer.indexOf('\n');
  if (newlineIndex === -1) {
    return null;
  }

  const line = buffer.slice(0, newlineIndex).trim();
  const rest = buffer.slice(newlineIndex + 1);
  return { line, rest };
}

// Helper function to output JSON messages to stdout
/*
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
*/

// Helper function to log status messages to stderr
function logStatus(message, level = 'info') {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] [${level.toUpperCase()}] ${message}\n`;
  process.stderr.write(logMessage);
}

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
    logStatus(`PID file checking skipped (not connecting to localhost)`);
    return true;
  }

  // Determine PID file location
  const pidFilePath = path.join(__dirname, '..', 'mcp_server.pid');

  if (fs.existsSync(pidFilePath)) {
    try {
      const pid = parseInt(fs.readFileSync(pidFilePath, 'utf8').trim());
      logStatus(`Found PID file with server process ID: ${pid}`);
      return true;
    } catch (error) {
      logStatus(`Found PID file but couldn't read it: ${error.message}`, 'warn');
      return false;
    }
  } else {
    logStatus(`No PID file found at ${pidFilePath}`, 'warn');
    logStatus(`Make sure the Python server is running before starting this bridge!`, 'warn');
    return true;
  }
}

// Define main MCP bridge function
async function startMcpBridge() {
  // Check if server appears to be running first
  if (!checkServerRunning()) {
    logStatus(`Server may not be running. Will attempt to connect anyway...`, 'warn');
  }

  // Construct server URL
  const serverUrl = `ws://${argv.host}:${argv.port}${argv.path}`;

  logStatus(`MCP Python Bridge starting...`);
  logStatus(`Connecting to Python MCP server at: ${serverUrl}`);

  // Set up standard input/output for passthrough
  process.stdin.setEncoding('utf8');

  // Create WebSocket client instance with increased frame size limits for large scan images
  const client = new WebSocketClient({
    maxReceivedFrameSize: 10 * 1024 * 1024, // 10MB (much larger than the ~1.69MB scan)
    maxReceivedMessageSize: 10 * 1024 * 1024 // 10MB for aggregate message size
  });

  // Handle connection failures
  client.on('connectFailed', (error) => {
    logStatus(`Connection error: ${error.toString()}`, 'error');
    process.exit(1);
  });

  // Handle successful connections
  client.on('connect', (connection) => {
    logStatus(`Connected to Python MCP server!`);
    logStatus(`Bridge ready! Passing messages between process and Python server...`);

    // Handle messages from the WebSocket
    connection.on('message', (message) => {
      try {
        if (message.type === 'utf8') {
          const wsData = message.utf8Data;

          // Attempt to parse as JSON (in case server sends JSON objects)
          try {
            // Parse and re-stringify to ensure proper JSON formatting
            const jsonData = JSON.parse(wsData);
            if (isHandshakeMessage(jsonData)) {
              logStatus(`Handshake WS->stdio ${describeJsonRpcMessage(jsonData)}`);
            }

            // MCP stdio uses newline-delimited JSON, not LSP Content-Length frames.
            writeStdioMessage(jsonData);
          } catch (parseError) {
            // Non-JSON websocket traffic is not valid MCP stdio; keep it off stdout.
            logStatus(
              `Ignoring non-JSON server message: ${parseError.message}; payload=${trimForLog(wsData)}`,
              'warn'
            );
          }
        }
              } catch (error) {
          logStatus(`Error processing server message: ${error.message}`, 'error');
        }
    });

    // Handle connection closure
    connection.on('close', (code, description) => {
      logStatus(`Connection to Python MCP server closed code=${code ?? 'unknown'} reason=${description || ''}`);
      process.exit(0);
    });

    // Handle connection errors
    connection.on('error', (error) => {
      logStatus(`Connection error: ${error.toString()}`, 'error');
    });

    // Buffer for collecting incoming data chunks
    let inputBuffer = '';

    // Set up stdin to forward to WebSocket
    process.stdin.on('data', (data) => {
      try {
        // Add the new data to our buffer
        inputBuffer += data.toString();

        while (true) {
          const framedMessage = tryConsumeStdioFrame(inputBuffer);
          if (framedMessage) {
            inputBuffer = framedMessage.rest;
            if (connection.connected) {
              try {
                const jsonInput = JSON.parse(framedMessage.body);
                if (isHandshakeMessage(jsonInput)) {
                  logStatus(`Handshake stdio->WS via content-length ${describeJsonRpcMessage(jsonInput)}`);
                }
                connection.sendUTF(JSON.stringify(jsonInput));
              } catch (jsonError) {
                logStatus(
                  `Failed to parse Content-Length stdin payload as JSON: ${jsonError.message}; payload=${trimForLog(framedMessage.body)}`,
                  'warn'
                );
                connection.sendUTF(framedMessage.body);
              }
            }
            continue;
          }

          const legacyMessage = tryConsumeLegacyLine(inputBuffer);
          if (!legacyMessage) {
            break;
          }

          inputBuffer = legacyMessage.rest;
          const line = legacyMessage.line;
          if (!line) {
            continue;
          }

          try {
            const jsonInput = JSON.parse(line);
            const textToSend = JSON.stringify(jsonInput);

            if (isHandshakeMessage(jsonInput)) {
              logStatus(`Handshake stdio->WS via ndjson ${describeJsonRpcMessage(jsonInput)}`);
            }

            if (connection.connected) {
              connection.sendUTF(textToSend);
            }
          } catch (jsonError) {
            logStatus(`Forwarding non-JSON stdin line to server: ${trimForLog(line)}`, 'warn');
            if (connection.connected) {
              connection.sendUTF(line);
            }
          }
        }
              } catch (error) {
          logStatus(`Error forwarding input to server: ${error.message}`, 'error');
        }
    });
  });

  // Handle process termination
  process.on('SIGINT', () => {
    logStatus(`Received SIGINT, shutting down MCP bridge...`);
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    logStatus(`Received SIGTERM, shutting down MCP bridge...`);
    process.exit(0);
  });

  // Connect to WebSocket server
  client.connect(serverUrl, 'mcp');
}

// Start the bridge
startMcpBridge().catch(error => {
  logStatus(`Fatal error: ${error.message}`, 'error');
  process.exit(1);
});
