#!/usr/bin/env python3
import asyncio
import logging
from mcp.client.session import ClientSession
from mcp.client.websocket import websocket_client
from mcp.types import TextContent, CallToolResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("mcp_client")
logger.setLevel(logging.DEBUG)

async def test_mcp_add():
    """Test the MCP addition server using the official MCP client with websocket transport."""
    server_url = "ws://127.0.0.1:8001"

    logger.info("🔌 CONNECTING TO SERVER: " + server_url + "/mcp")

    try:
        # Connect to the MCP server using the official websocket client
        async with websocket_client(server_url + "/mcp") as streams:
            logger.info("✅ CONNECTED!")

            # Create a ClientSession to communicate with the server
            async with ClientSession(*streams) as session:
                # Initialize the session
                logger.info("🚀 INITIALIZING SESSION")
                init_result = await session.initialize()
                logger.info(f"📥 INITIALIZED SESSION WITH: {init_result.serverInfo.name}")

                # List available tools
                logger.info("🔍 LISTING AVAILABLE TOOLS")
                tools_result = await session.list_tools()

                # Print available tools
                logger.info(f"📦 AVAILABLE TOOLS ({len(tools_result.tools)}):")
                for tool in tools_result.tools:
                    logger.info(f"   🔧 {tool.name}: {tool.description}")

                # --- Test dynamic function registration --- START ---
                logger.info("🔧 REGISTERING A DYNAMIC 'greet' FUNCTION (SCHEMA AUTO-DETECTED)")
                greet_code = """
def greet(name: str, times: int = 1) -> str:
    \"\"\"Greets a person multiple times.\"\"\"
    greetings = []
    for _ in range(times):
        greetings.append(f'Hello, {name}!')
    return ' '.join(greetings)
"""
                register_args = {
                    "name": "greet",
                    "code": greet_code,
                    "description": "A dynamically registered greeting function"
                    # No input_schema provided!
                }
                try:
                    register_result = await session.call_tool("register_function", register_args)
                    for item in register_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ REGISTRATION RESULT: {item.text}")
                except Exception as e:
                    logger.error(f"❌ FAILED TO REGISTER FUNCTION: {str(e)}")

                # List tools again to see if the new one appears
                logger.info("🔍 LISTING AVAILABLE TOOLS AGAIN")
                tools_result_after_register = await session.list_tools()
                logger.info(f"📦 AVAILABLE TOOLS ({len(tools_result_after_register.tools)}):")
                for tool in tools_result_after_register.tools:
                    logger.info(f"   🔧 {tool.name}: {tool.description}")
                # --- Test dynamic function registration --- END ---

                # Call the add tool with two numbers
                a, b = 5, 7
                logger.info(f"🧮 CALLING ADD TOOL: {a} + {b}")
                call_result = await session.call_tool(
                    "add",
                    {"a": a, "b": b}
                )

                # Display the result - call_result is a CallToolResult type
                for item in call_result.content:
                    if isinstance(item, TextContent):
                        result = item.text
                        logger.info(f"✨ ADDITION RESULT: {a} + {b} = {result}")
                        print(f"\n✨ ADDITION RESULT: {a} + {b} = {result} ✨\n")

                # Only the client connection is disconnecting, the server remains running
                logger.info("👋 CLIENT DISCONNECTING (SERVER REMAINS ACTIVE FOR FUTURE CONNECTIONS)")
                # The MCP server will stay running for long-running conversations and future connections

    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    print("\n🐱 MCP WEBSOCKET TEST CLIENT 🐱")
    print("==============================")
    asyncio.run(test_mcp_add())
