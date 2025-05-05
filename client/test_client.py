#!/usr/bin/env python3
import asyncio
import argparse
import logging
import json
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

                # --- ADDED: Define and Register 'bar' function ---
                # Step 1: Add the empty 'bar' function stub
                logger.info("🔧 ADDING 'bar' FUNCTION STUB VIA _function_add")
                add_bar_args = {"name": "bar"}
                try:
                    add_bar_result = await session.call_tool("_function_add", add_bar_args)
                    for item in add_bar_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ ADD BAR STUB RESULT: {item.text}")
                except Exception as e:
                    logger.error(f"❌ FAILED TO ADD 'bar' STUB: {str(e)}")

                # Step 2: Set the code for the 'bar' function
                logger.info("🔧 REGISTERING A 'bar' FUNCTION VIA _function_set")
                bar_code = """
def bar(x: int, y: int) -> int:
    \"\"\"Subtracts integer y from integer x.\"\"\"
    result = x - y
    print(f'[bar tool execution] {x} - {y} = {result}') # Add server-side print
    return result
"""
                set_bar_args = {"code": bar_code}
                try:
                    set_bar_result = await session.call_tool("_function_set", set_bar_args)
                    for item in set_bar_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ SET BAR FUNCTION RESULT: {item.text}")
                except Exception as e:
                    logger.error(f"❌ FAILED TO SET 'bar' FUNCTION: {str(e)}")
                # --- END Define and Register 'bar' function ---

                # --- ADDED: Get and print code for foo ---
                logger.info("📄 GETTING CODE FOR 'foo' VIA _function_get")
                get_foo_args = {"name": "foo"}
                try:
                    get_foo_result = await session.call_tool("_function_get", get_foo_args)
                    for item in get_foo_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ GOT FOO CODE:\n-------\n{item.text}\n-------")
                except Exception as e:
                    logger.error(f"❌ FAILED TO GET 'foo' CODE: {str(e)}")
                # --- END Get and print code for foo ---

                # --- ADDED: Get and print code for bar ---
                logger.info("📄 GETTING CODE FOR 'bar' VIA _function_get")
                get_bar_args = {"name": "bar"}
                try:
                    get_bar_result = await session.call_tool("_function_get", get_bar_args)
                    for item in get_bar_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ GOT BAR CODE:\n-------\n{item.text}\n-------")
                except Exception as e:
                    logger.error(f"❌ FAILED TO GET 'bar' CODE: {str(e)}")
                # --- END Get and print code for bar ---

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

                # --- ADDED: Call the bar tool ---
                x_bar, y_bar = 10, 3
                logger.info(f"🧮 CALLING BAR TOOL: {x_bar} - {y_bar}")
                try:
                    call_bar_result = await session.call_tool(
                        "bar",
                        {
                            "x": x_bar,
                            "y": y_bar
                        }
                    )
                    # Display the result
                    for item in call_bar_result.content:
                        if isinstance(item, TextContent):
                            result = item.text
                            logger.info(f"✨ BAR RESULT: {x_bar} - {y_bar} = {result}")
                            print(f"\n✨ BAR RESULT: {x_bar} - {y_bar} = {result} ✨\n")
                except Exception as e:
                     logger.error(f"❌ FAILED TO CALL 'bar' FUNCTION: {str(e)}")
                # --- END Call the bar tool ---

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

                # --- ADDED: Remove 'bar' function ---
                logger.info("🗑️ REMOVING 'bar' FUNCTION VIA _function_remove")
                remove_bar_args = {"name": "bar"}
                try:
                    remove_bar_result = await session.call_tool("_function_remove", remove_bar_args)
                    for item in remove_bar_result.content:
                        if isinstance(item, TextContent):
                            logger.info(f"✅ REMOVE BAR RESULT: {item.text}")
                except Exception as e:
                    logger.error(f"❌ FAILED TO REMOVE 'bar' FUNCTION: {str(e)}")
                # --- END REMOVE 'bar' function ---

                # --- Test dynamic server management --- START ---
                server_name = "test_sleep_server"
                server_config = {
                    "mcpServers": {
                        server_name: {
                            "command": "sleep",
                            "args": ["60"] # Sleep for 60 seconds
                        }
                    }
                }
                logger.info(f"🧪 TESTING SERVER MANAGEMENT FOR: {server_name}")

                try:
                    # Step 1: Add the server config
                    logger.info(f"🔧 ADDING SERVER CONFIG: {server_name}")
                    add_server_args = {"name": server_name, "config": server_config}
                    try:
                        add_server_result = await session.call_tool("_server_add", add_server_args)
                        for item in add_server_result.content:
                            if isinstance(item, TextContent):
                                logger.info(f"✅ ADD SERVER RESULT: {item.text}")
                                # Basic check - more robust tests could assert content
                                assert "added" in item.text.lower() or "updated" in item.text.lower()
                    except Exception as e:
                        logger.error(f"❌ FAILED TO ADD SERVER '{server_name}': {str(e)}")
                        raise # Re-raise to indicate test failure in this block

                    # Step 2: Start the server
                    logger.info(f"🚀 STARTING SERVER: {server_name}")
                    start_server_args = {"name": server_name}
                    try:
                        start_server_result = await session.call_tool("_server_start", start_server_args)
                        logger.info(f"✅ START SERVER RESULT: {start_server_result}")
                        assert isinstance(start_server_result, CallToolResult)
                        # Optional: Add list_tools check here for runningStatus/started_at
                        logger.info("⏳ Waiting 3 seconds for server state to propagate...")
                        await asyncio.sleep(3) # <<< INCREASED SLEEP DURATION

                        # ---> Verify server status in tool list after start <--- #
                        logger.info(f"🔍 VERIFYING TOOL LIST STATUS for {server_name} after START")
                        # --- MODIFIED: Call _server_get_status instead of list_tools ---
                        status_result_start = await session.call_tool('_server_get_status', {'name': server_name})
                        logger.info(f"✅ SERVER STATUS RESULT (After Start): {status_result_start}")
                        assert isinstance(status_result_start, CallToolResult)
                        assert status_result_start is not None and len(status_result_start) > 0
                        status_data_start = json.loads(status_result_start[0].text)
                        logger.info(f"📊 GOT SERVER STATUS (after start): {status_data_start}")

                        assert status_data_start.get('runningStatus') == 'running', f"Server '{server_name}' not 'running' after start"
                        assert status_data_start.get('started_at') is not None, f"Server '{server_name}' started_at is None after start"
                        logger.info(f"✅ Server '{server_name}' has runningStatus='running' and started_at is set.")
                        # -------------------------------------------------------------------

                        # --------------------
                        # --- STOP SERVER --- #
                    except Exception as e:
                        logger.error(f"❌ FAILED TO START SERVER '{server_name}': {str(e)}")
                        raise

                    # Step 3: Stop the server
                    logger.info(f"🛑 STOPPING SERVER: {server_name}")
                    stop_server_args = {"name": server_name}
                    try:
                        stop_server_result = await session.call_tool("_server_stop", stop_server_args)
                        logger.info(f"✅ STOP SERVER RESULT: {stop_server_result}")
                        assert isinstance(stop_server_result, CallToolResult)
                        # Optional: Add list_tools check here for runningStatus/started_at
                        await asyncio.sleep(1) # Give it a moment to register stopped state

                        # ---> Verify server status in tool list after stop <--- #
                        logger.info(f"🔍 VERIFYING TOOL LIST STATUS for {server_name} after STOP")
                        # --- MODIFIED: Call _server_get_status instead of list_tools ---
                        status_result_stop = await session.call_tool('_server_get_status', {'name': server_name})
                        logger.info(f"✅ SERVER STATUS RESULT (After Stop): {status_result_stop}")
                        assert isinstance(status_result_stop, CallToolResult)
                        assert status_result_stop is not None and len(status_result_stop) > 0
                        status_data_stop = json.loads(status_result_stop[0].text)
                        logger.info(f"📊 GOT SERVER STATUS (after stop): {status_data_stop}")

                        assert status_data_stop.get('runningStatus') == 'stopped', f"Server '{server_name}' not 'stopped' after stop"
                        assert status_data_stop.get('started_at') is None, f"Server '{server_name}' started_at is not None after stop"
                        logger.info(f"✅ Server '{server_name}' has runningStatus='stopped' and started_at is None.")
                        # -------------------------------------------------------------------

                        # -----------------------
                        # --- REMOVE SERVER --- #
                    except Exception as e:
                        logger.error(f"❌ FAILED TO STOP SERVER '{server_name}': {str(e)}")
                        raise
                finally:
                    # Step 4: Clean up - Remove the server config
                    logger.info(f"🗑️ REMOVING SERVER CONFIG (Cleanup): {server_name}")
                    remove_server_args = {"name": server_name}
                    try:
                        remove_server_result = await session.call_tool("_server_remove", remove_server_args)
                        logger.info(f"✅ REMOVE SERVER RESULT: {remove_server_result}")
                        assert isinstance(remove_server_result, CallToolResult)
                    except Exception as e:
                        # Log cleanup error but don't fail the test run if main part succeeded
                        logger.error(f"⚠️ FAILED TO REMOVE SERVER '{server_name}' during cleanup: {str(e)}")

                logger.info(f"✅ SERVER MANAGEMENT TEST COMPLETE FOR: {server_name}")
                # --- Test dynamic server management --- END ---

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
    parser.add_argument('--port', type=int, default=8000, help='Port number of the MCP server to connect to.')
    args = parser.parse_args()
    server_url = f'ws://127.0.0.1:{args.port}'
    asyncio.run(test_mcp_add(server_url))
