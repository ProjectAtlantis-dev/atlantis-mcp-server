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
    server_url = "ws://127.0.0.1:8000"
    
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
