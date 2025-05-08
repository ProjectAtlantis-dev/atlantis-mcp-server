

import os
import inspect
import asyncio
import json
import logging
import pathlib
import shutil
import datetime
from typing import Any, Dict, Optional, List, Tuple, Union

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import TextContent, Tool, ListToolsResult, ListToolsRequest

from state import (
    logger
)

class EmCeePee:
    """
    An MCP task, may or may not be running
    """

    def __init__(self, name:str, servers_dir: str):
        self.servers_dir = servers_dir
        self.name = name
        self.safe_name = f"{name}.json"

        self.load_error = None
        self.start_time = None

        self.config = None
        self.state = "init"

    async def _handle_load_error(self, name: str, error_msg: str):
        # Get the caller function's name
        caller_frame = inspect.currentframe().f_back
        caller_name = caller_frame.f_code.co_name if caller_frame else "unknown"

        logger.error(f"❌ {caller_name}: {error_msg}")
        self.load_error = error_msg
        await self._write_server_error_log(name, error_msg)
        self.state = "error"
        self.config = None

    async def _fs_load(self, name: str):
        """
        Loads the config for server '{name}' from {name}.json in servers_dir.
        Returns the parsed JSON dict on success.
        Returns the raw file content (str) if JSON parsing fails.
        Returns None if the file doesn't exist or an IO error occurs.
        """
        safe_name = f"{name}.json"
        file_path = os.path.join(self.servers_dir, safe_name)

        try :
            if not os.path.exists(file_path):
                raise Exception("Existing config file not found for '{name}' at {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            # Attempt to parse the JSON
            self.config_data = json.loads(raw_content)

            # Success: clear any cached error for this server
            self._server_load_error = None
            self.state = "loaded"

        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}"
            logger.error(f"❌ _fs_load: {error_msg}")
            self._server_load_error = error_msg  # Cache the error
            await self._write_server_error_log(name, error_msg, raw_content)
            self.state = "error"

        except IOError as e:
            error_msg = f"IO error reading {file_path}: {e}"
            logger.error(f"❌ _fs_load: {error_msg}")
            self._server_load_error = error_msg  # Cache the error
            await self._write_server_error_log(name, error_msg)
            self.state = "error"

        except Exception as e:
            error_msg = f"Unexpected error loading {file_path}: {e}"
            logger.error(f"❌ _fs_load: {error_msg}", exc_info=True)
            self._server_load_error = error_msg  # Cache the error
            await self._write_server_error_log(name, error_msg)
            self.state = "error"


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

    async def _fs_save(self, config: Dict[str, Any]):
            """
            Saves the JSON config dict to {name}.json in servers_dir.
            Returns the full path if successful, None otherwise.
            """
            file_path = os.path.join(self.servers_dir, self.safe_name)
            logger.debug(f"---> _fs_save: Attempting to save '{self.name}' to path: {file_path}")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"💾 Saved server config for '{self.name}' to {file_path}")

            except Exception as e:
                error_msg = f"Unexpected error writing {file_path}: {e}"
                logger.error(f"❌ _fs_save: {error_msg}", exc_info=True)
                self._server_load_error = error_msg  # Cache the error
                await self._write_server_error_log(self.name, error_msg)
                self.state = "error"


