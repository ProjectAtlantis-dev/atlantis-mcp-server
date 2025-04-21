#!/usr/bin/env python3
import asyncio
import argparse
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

async def test_mcp_add(server_url: str):
    """Test the MCP addition server using the official MCP client with websocket transport."""

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
                    # Access type from annotations dictionary via __pydantic_extra__
                    tool_type = tool.__pydantic_extra__.get('annotations', {}).get('type', 'internal') # Default to internal if missing
                    logger.info(f"   🔧 [{tool_type}] {tool.name}: {tool.description}") 

                # --- Test dynamic function registration --- START ---
                # Step 1: Add the empty 'greet' function stub
                logger.info("🔧 ADDING 'greet' FUNCTION STUB VIA _function_add")
                add_greet_args = {"name": "greet"}
                try:
                    add_greet_result = await session.call_tool("_function_add", add_greet_args)
                    for item in add_greet_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ ADD GREET STUB RESULT: {item.text}")
                except Exception as e:
                    logger.error(f"❌ FAILED TO ADD 'greet' STUB: {str(e)}")

                # Step 2: Set the code for the 'greet' function
                logger.info("🔧 REGISTERING A DYNAMIC 'greet' FUNCTION VIA _function_set")
                greet_code = """
def greet(name: str, times: int = 1) -> str:
    \"\"\"Greets a person multiple times.\"\"\"
    greetings = []
    for _ in range(times):
        greetings.append(f'Hello, {name}!')
    return ' '.join(greetings)
"""
                # _function_set takes only the code; name/schema are derived
                set_args = {"code": greet_code}
                try:
                    # Call the correct tool name
                    set_result = await session.call_tool("_function_set", set_args)
                    for item in set_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ SET FUNCTION RESULT: {item.text}")
                except Exception as e:
                    logger.error(f"❌ FAILED TO SET 'greet' FUNCTION: {str(e)}")

                # --- ADDED: Define and Register 'foo' function --- 
                # Step 1: Add the empty 'foo' function stub
                logger.info("🔧 ADDING 'foo' FUNCTION STUB VIA _function_add")
                add_foo_args = {"name": "foo"}
                try:
                    add_foo_result = await session.call_tool("_function_add", add_foo_args)
                    for item in add_foo_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ ADD FOO STUB RESULT: {item.text}")
                except Exception as e:
                    logger.error(f"❌ FAILED TO ADD 'foo' STUB: {str(e)}")

                # Step 2: Set the code for the 'foo' function
                logger.info("🔧 REGISTERING A 'foo' FUNCTION VIA _function_set")
                foo_code = """
def foo(a: int, b: int) -> int:
    \"\"\"Adds two integers together (renamed from add).\"\"\"
    result = a + b
    print(f'[foo tool execution] {a} + {b} = {result}') # Add server-side print
    return result
"""
                set_foo_args = {"code": foo_code}
                try:
                    set_foo_result = await session.call_tool("_function_set", set_foo_args)
                    for item in set_foo_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ SET FOO FUNCTION RESULT: {item.text}")
                except Exception as e:
                    logger.error(f"❌ FAILED TO SET 'foo' FUNCTION: {str(e)}")

                # List tools again to see if the new one appears
                logger.info("🔍 LISTING AVAILABLE TOOLS AGAIN")
                tools_result_after_register = await session.list_tools()
                logger.info(f"📦 AVAILABLE TOOLS ({len(tools_result_after_register.tools)}):")
                for tool in tools_result_after_register.tools:
                    # Access type from annotations dictionary via __pydantic_extra__
                    tool_type = tool.__pydantic_extra__.get('annotations', {}).get('type', 'internal') # Default to internal if missing
                    logger.info(f"   🔧 [{tool_type}] {tool.name}: {tool.description}")

                # Call the foo tool with two numbers
                a_foo, b_foo = 5, 7
                logger.info(f"🧮 CALLING FOO TOOL: {a_foo} + {b_foo}")
                call_result = await session.call_tool(
                    "foo",
                    {
                        "a": a_foo,
                        "b": b_foo
                    }
                )

                # Display the result - call_result is a CallToolResult type
                for item in call_result.content:
                    if isinstance(item, TextContent):
                        result = item.text
                        logger.info(f"✨ FOO RESULT: {a_foo} + {b_foo} = {result}")
                        print(f"\n✨ FOO RESULT: {a_foo} + {b_foo} = {result} ✨\n")

                # --- ADDED: Remove 'foo' function ---
                logger.info("🗑️ REMOVING 'foo' FUNCTION VIA _function_remove")
                remove_foo_args = {"name": "foo"}
                try:
                    remove_foo_result = await session.call_tool("_function_remove", remove_foo_args)
                    for item in remove_foo_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ REMOVE FOO RESULT: {item.text}")
                except Exception as e:
                    logger.error(f"❌ FAILED TO REMOVE 'foo' FUNCTION: {str(e)}")

                # List tools one last time to confirm removal
                logger.info("🔍 LISTING AVAILABLE TOOLS AFTER REMOVAL")
                tools_result_after_remove = await session.list_tools()
                logger.info(f"📦 AVAILABLE TOOLS ({len(tools_result_after_remove.tools)}):")
                for tool in tools_result_after_remove.tools:
                    # Access type from annotations dictionary via __pydantic_extra__
                    tool_type = tool.__pydantic_extra__.get('annotations', {}).get('type', 'internal') # Default to internal if missing
                    logger.info(f"   🔧 [{tool_type}] {tool.name}: {tool.description}")

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
    parser = argparse.ArgumentParser(description='MCP Test Client')
    parser.add_argument('--port', type=int, default=8002, help='Port number of the MCP server to connect to.')
    args = parser.parse_args()
    server_url = f'ws://127.0.0.1:{args.port}'
    asyncio.run(test_mcp_add(server_url))
