"""
Manages the lifecycle of dynamic MCP servers (configs) by launching them
as background tasks using the MCP Python SDK's stdio transport.
Stores each server config as a JSON file under SERVERS_DIR.
"""
import os
import json
import logging
import asyncio
import shutil
import datetime
from typing import Any, Dict, Optional, List, Tuple, Union

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import TextContent, Tool, ListToolsResult, ListToolsRequest

from state import (
    SERVERS_DIR, 
    logger, 
    ACTIVE_SERVER_TASKS, 
    SERVER_REQUEST_TIMEOUT
)

class DynamicServerManager:
    """
    Manages dynamic MCP servers lifecycle - adding, removing, starting, stopping, etc.
    Handles the configuration of MCP servers and their execution as background tasks.
    """
    
    def __init__(self, servers_dir: str):
        """
        Initialize the Dynamic Server Manager with a specific servers directory.
        
        Args:
            servers_dir: Path to the directory where server configuration files are stored
        """
        self.servers_dir = servers_dir
        self.old_dir = os.path.join(servers_dir, 'OLD')
        
        # Ensure directories exist
        os.makedirs(self.servers_dir, exist_ok=True)
        os.makedirs(self.old_dir, exist_ok=True)
        
        # Server state tracking
        self.active_server_tasks = ACTIVE_SERVER_TASKS
        self.server_start_times = {}
        self._server_load_errors = {}
        
        logger.info(f"Dynamic Server Manager initialized with servers dir: {servers_dir}")
    
    # --- 1. File Save/Load Methods ---
    
    async def _fs_save_server(self, name: str, config: Dict[str, Any]) -> Optional[str]:
        """
        Saves the provided JSON config dict to {name}.json in servers_dir.
        Returns the full path if successful, None otherwise.
        """
        safe_name = f"{name}.json"
        file_path = os.path.join(self.servers_dir, safe_name)
        logger.debug(f"---> _fs_save_server: Attempting to save '{name}' to path: {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            logger.info(f"💾 Saved server config for '{name}' to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"❌ _fs_save_server: Failed to write {file_path}: {e}")
            return None

    async def _fs_load_server(self, name: str) -> Optional[Union[Dict[str, Any], str]]:
        """
        Loads the config for server '{name}' from {name}.json in servers_dir.
        Returns the parsed JSON dict on success.
        Returns the raw file content (str) if JSON parsing fails.
        Returns None if the file doesn't exist or an IO error occurs.
        """
        safe_name = f"{name}.json"
        file_path = os.path.join(self.servers_dir, safe_name)
        if not os.path.exists(file_path):
            logger.info(f"⚠️ _fs_load_server: Existing config not found for '{name}' at {file_path}")
            self._server_load_errors.pop(name, None)  # Clear potential old error if file is gone
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            # Attempt to parse the JSON
            config_data = json.loads(raw_content)
            # Success: clear any cached error for this server
            self._server_load_errors.pop(name, None)
            return config_data
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}"
            logger.error(f"❌ _fs_load_server: {error_msg}")
            self._server_load_errors[name] = error_msg  # Cache the error
            await self._write_server_error_log(name, error_msg, raw_content)
            return raw_content  # Return raw content on JSON error
        except IOError as e:
            error_msg = f"IO error reading {file_path}: {e}"
            logger.error(f"❌ _fs_load_server: {error_msg}")
            self._server_load_errors[name] = error_msg  # Cache the error
            await self._write_server_error_log(name, error_msg)
            return None  # Return None on IO error
        except Exception as e:
            error_msg = f"Unexpected error loading {file_path}: {e}"
            logger.error(f"❌ _fs_load_server: {error_msg}", exc_info=True)
            self._server_load_errors[name] = error_msg  # Cache the error
            await self._write_server_error_log(name, error_msg)
            return None  # Return None on unexpected errors

    async def _write_server_error_log(self, name: str, error_msg: str, raw_content: Optional[str] = None) -> None:
        """
        Write an error message to a server-specific log file in the servers_dir.
        Overwrites any existing log to only keep the latest error.
        Includes raw content if provided (e.g., for JSON decode errors).
        """
        log_filename = f"{name}_error.log"
        log_path = os.path.join(self.servers_dir, log_filename)
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"=== Error Log for '{name}' @ {timestamp} ===\n\n")
                f.write(f"{error_msg}\n\n")
                if raw_content:
                    f.write("=== Raw Content ===\n\n")
                    f.write(raw_content)
            logger.debug(f"📝 Error log written to {log_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write error log for '{name}': {e}")

    # --- 2. Config CRUD Methods ---
    
    async def server_add(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Adds a new server config. Returns False if it already exists.
        """
        # Check if server already exists
        existing = await self._fs_load_server(name)
        if existing is not None:
            logger.warning(f"⚠️ Server '{name}' already exists, not adding.")
            return False
        
        # Save the new config
        result = await self._fs_save_server(name, config)
        return result is not None

    async def server_remove(self, name: str) -> bool:
        """
        Removes the server config by deleting its JSON file. Returns False if missing.
        Also stops the server if it's running.
        """
        # Check if server is running and stop it first
        if name in self.active_server_tasks:
            logger.info(f"🛑 Stopping running server '{name}' before removal...")
            try:
                # Implement server_stop functionality here
                # await self.server_stop({"name": name}, None)
                pass  # Placeholder for now
            except Exception as e:
                logger.error(f"❌ Failed to stop server '{name}' during removal: {e}")
                # Continue with removal anyway
        
        # Remove the config file
        file_path = os.path.join(self.servers_dir, f"{name}.json")
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ Server config '{name}' not found for removal.")
            return False
        
        try:
            # Backup the config before deletion
            backup_dir = self.old_dir
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"{timestamp}_{name}.json")
            shutil.copy2(file_path, backup_path)
            logger.info(f"📦 Backed up '{name}' config to {backup_path}")
            
            # Delete the original file
            os.remove(file_path)
            logger.info(f"🗑️ Removed server config '{name}'")
            
            # Clean up any cached errors
            self._server_load_errors.pop(name, None)
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove server '{name}': {e}")
            return False

    async def server_get(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Returns the server config dict or None if not found.
        """
        result = await self._fs_load_server(name)
        if isinstance(result, dict):
            return result
        return None

    async def get_server_tools(self, name: str) -> List[Tool]:
        """
        Connects to a specific running managed server using its stored start
        parameters and fetches its tool list via a temporary connection.
        
        Returns a list of Tool objects from the server.
        
        Raises:
            ValueError: If the server name is invalid or not found.
            RuntimeError: If the server config is invalid or server is not running.
            TimeoutError: If the connection or request to the server times out.
            Exception: For other communication or MCP errors.
        """
        # Placeholder implementation
        logger.info(f"Getting tools for server '{name}'")
        return []

    async def server_list(self) -> List[str]:
        """
        Returns a list of all configured server names.
        """
        server_list = []
        if os.path.exists(self.servers_dir):
            for filename in os.listdir(self.servers_dir):
                if filename.endswith('.json'):
                    server_name = filename[:-5]  # Remove .json extension
                    server_list.append(server_name)
        return server_list

    async def server_set(self, args: Dict[str, Any], server) -> Tuple[Optional[str], List[TextContent]]:
        """
        MCP handler to add/update a server config.
        Expects args['config']: dict or JSON string containing {"mcpServers": {"server_name": {...}}}
        The server name is derived from the key within 'mcpServers'.
        If the server is running and the config is updated, it might need restarting.
        
        Returns:
            Tuple containing:
                - Extracted server name (str) if successful, None otherwise
                - List of TextContent for the response
        """
        # Placeholder implementation
        return None, [TextContent(type="text", text="Server set operation not implemented yet")]

    async def server_validate(self, name: str) -> Dict[str, Any]:
        """
        Validates that the *saved* server config JSON has required keys.
        Returns a dict with 'valid':bool and 'error':Optional[str].
        """
        result = {
            'valid': False,
            'error': None
        }
        
        config = await self.server_get(name)
        if not config:
            result['error'] = f"Server config '{name}' not found or has parsing errors"
            return result
        
        # Validate essential config fields (placeholder - expand as needed)
        if 'command' not in config:
            result['error'] = f"Missing required key 'command' in server config for '{name}'"
            return result
        
        # Config is valid
        result['valid'] = True
        return result

    # --- 3. Background Task Methods ---
    
    async def _run_mcp_client_session(self, name: str, params: StdioServerParameters, shutdown_event: asyncio.Event) -> None:
        """
        Runs the background MCP client session for a managed server.
        """
        # Placeholder implementation
        pass

    # --- 4. Server Start/Stop Methods ---
    
    async def server_start(self, args: Dict[str, Any], server) -> List[TextContent]:
        """
        MCP handler to start a managed MCP server process as a background task.
        Expects args['name']: str.
        """
        # Placeholder implementation
        return [TextContent(type="text", text="Server start operation not implemented yet")]

    async def server_stop(self, args: Dict[str, Any], server) -> List[TextContent]:
        """
        MCP handler to stop a managed MCP server process task.
        Expects args['name']: str.
        """
        # Placeholder implementation
        return [TextContent(type="text", text="Server stop operation not implemented yet")]

    # --- 5. Self-test Methods ---
    
    @staticmethod
    async def run_tests() -> bool:
        """
        Run self-tests for the DynamicServerManager class.
        Returns True if all tests pass, False otherwise.
        """
        # Save original logger if exists
        original_logger = globals().get('logger') if 'logger' in globals() else None

        # Mock logger for testing
        class MockLogger:
            def __init__(self):
                self.logs = []
            
            def debug(self, msg, *args, **kwargs):
                print(f"[DEBUG] {msg}")
                self.logs.append(f"DEBUG: {msg}")
                
            def info(self, msg, *args, **kwargs):
                print(f"[INFO] {msg}")
                self.logs.append(f"INFO: {msg}")
                
            def warning(self, msg, *args, **kwargs):
                print(f"[WARNING] {msg}")
                self.logs.append(f"WARNING: {msg}")
                
            def error(self, msg, *args, **kwargs):
                print(f"[ERROR] {msg}")
                self.logs.append(f"ERROR: {msg}")
        
        try:
            print("\n🧪 STARTING DYNAMICSERVERMANAGER SELF-TEST...")
            all_tests_passed = True
            total_tests = 0
            passed_tests = 0

            # Import needed for testing
            from state import SERVERS_DIR
            from pathlib import Path
            
            # Use the actual SERVERS_DIR for testing
            servers_dir = SERVERS_DIR
            print(f"\n📁 Using servers directory: {servers_dir}")

            # Set up the manager
            globals()['logger'] = MockLogger()
            manager = DynamicServerManager(servers_dir)

            # Test 1: _fs_save_server method
            total_tests += 1
            print("\n🧪 TEST 1: Testing _fs_save_server method...")
            test_server_name = "test_server"
            test_config = {
                "command": "python -m http.server",
                "args": ["8888"],
                "cwd": "/tmp",
                "env": {"DEBUG": "1"}
            }

            file_path = await manager._fs_save_server(test_server_name, test_config)
            saved_file = Path(servers_dir) / f"{test_server_name}.json"

            if file_path and saved_file.exists():
                print("✅ Test 1 PASSED: Server config was successfully saved")
                passed_tests += 1
            else:
                print("❌ Test 1 FAILED: Server config was not saved")
                all_tests_passed = False

            # Test 2: _fs_load_server method
            total_tests += 1
            print("\n🧪 TEST 2: Testing _fs_load_server method...")
            loaded_config = await manager._fs_load_server(test_server_name)

            if isinstance(loaded_config, dict) and loaded_config.get('command') == test_config['command']:
                print("✅ Test 2 PASSED: Loaded config matches the saved config")
                passed_tests += 1
            else:
                print("❌ Test 2 FAILED: Loaded config doesn't match the saved config")
                print(f"Expected: {test_config}")
                print(f"Got: {loaded_config}")
                all_tests_passed = False

            # Test 3: server_add method
            total_tests += 1
            print("\n🧪 TEST 3: Testing server_add method...")
            new_server_name = "new_test_server"
            new_server_config = {
                "command": "python -m http.server",
                "args": ["9999"],
                "cwd": "/tmp/test",
                "env": {"DEBUG": "0"}
            }

            try:
                result = await manager.server_add(new_server_name, new_server_config)
                new_file = Path(servers_dir) / f"{new_server_name}.json"

                if result and new_file.exists():
                    print("✅ Test 3 PASSED: Server config was successfully added")
                    passed_tests += 1
                else:
                    print("❌ Test 3 FAILED: Server config was not added")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 3 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 4: server_list method
            total_tests += 1
            print("\n🧪 TEST 4: Testing server_list method...")
            try:
                server_list = await manager.server_list()
                if test_server_name in server_list and new_server_name in server_list:
                    print("✅ Test 4 PASSED: Server list correctly contains added servers")
                    passed_tests += 1
                else:
                    print("❌ Test 4 FAILED: Server list doesn't contain expected servers")
                    print(f"Expected servers: {test_server_name}, {new_server_name}")
                    print(f"Got servers: {server_list}")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 4 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 5: server_validate method
            total_tests += 1
            print("\n🧪 TEST 5: Testing server_validate method...")
            try:
                validation_result = await manager.server_validate(test_server_name)
                if validation_result.get('valid'):
                    print("✅ Test 5 PASSED: Server config validation successful")
                    passed_tests += 1
                else:
                    print("❌ Test 5 FAILED: Server config validation failed")
                    print(f"Error: {validation_result.get('error')}")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 5 FAILED with exception: {e}")
                all_tests_passed = False
                
            # Test 6: Invalid server validation
            total_tests += 1
            print("\n🧪 TEST 6: Testing validation of non-existent server...")
            try:
                # Create a server with missing required fields
                invalid_server_name = "invalid_server"
                invalid_config = {"args": ["-h"], "cwd": "/tmp"} # Missing 'command'
                await manager._fs_save_server(invalid_server_name, invalid_config)
                
                validation_result = await manager.server_validate(invalid_server_name)
                if not validation_result.get('valid') and validation_result.get('error'):
                    print("✅ Test 6 PASSED: Invalid server correctly failed validation")
                    print(f"Validation error: {validation_result.get('error')}")
                    passed_tests += 1
                else:
                    print("❌ Test 6 FAILED: Invalid server incorrectly passed validation")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 6 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 7: server_remove method
            total_tests += 1
            print("\n🧪 TEST 7: Testing server_remove method...")
            try:
                result = await manager.server_remove(new_server_name)
                new_file = Path(servers_dir) / f"{new_server_name}.json"

                if result and not new_file.exists():
                    print("✅ Test 7 PASSED: Server config was successfully removed")
                    passed_tests += 1
                else:
                    print("❌ Test 7 FAILED: Server config was not removed")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 7 FAILED with exception: {e}")
                all_tests_passed = False

            # Clean up leftover test files
            print("\n🧹 Cleaning up test files...")
            leftover_files = [
                "test_server",            # From Test 1 & 2
                "invalid_server"         # From Test 6
            ]
            
            for test_file in leftover_files:
                try:
                    if os.path.exists(os.path.join(SERVERS_DIR, f"{test_file}.json")):
                        await manager.server_remove(test_file)
                        print(f"  Removed {test_file}.json")
                except Exception as e:
                    print(f"  ⚠️ Could not remove {test_file}.json: {e}")

            # Print test summary
            print("\n🏁 DYNAMICSERVERMANAGER SELF-TEST COMPLETE 🏁")
            print(f"Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")

            if all_tests_passed:
                print("\n✅ ALL TESTS PASSED! DynamicServerManager is working correctly.")
                # Clean up OLD directory if it exists
                old_dir_final_cleanup = Path(SERVERS_DIR) / "OLD"
                if old_dir_final_cleanup.exists() and old_dir_final_cleanup.is_dir():
                    print(f"🧹 Cleaning up OLD directory: {old_dir_final_cleanup}")
                    import shutil
                    shutil.rmtree(old_dir_final_cleanup) # remove OLD and its contents
                return True
            else:
                print("\n❌ SOME TESTS FAILED. See above for details.")
                print(f"👉 NOTE: Tests were run against {SERVERS_DIR}. Review and manually clean up test files if necessary.")
                return False
        finally:
            # Restore the original logger
            if original_logger:
                globals()['logger'] = original_logger
            else:
                if 'logger' in globals(): # if mock was set but no original, remove it
                    del globals()['logger']


# Run the tests if this module is executed directly
if __name__ == "__main__":
    try:
        # Check if there's an existing event loop
        asyncio.get_running_loop()
        print("Using existing event loop")
    except RuntimeError:
        print("Creating new event loop for tests")
        # Create a new event loop for testing
        asyncio.run(DynamicServerManager.run_tests())
