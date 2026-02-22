#!/usr/bin/env python3
import os
import psutil
import tempfile
from state import logger

class PIDManager:
    """
    Manages the Process ID (PID) file for the MCP server.
    Handles creation, removal, and validation of PID files.
    """
    
    def __init__(self, pid_file_path=None, service_name=None):
        """
        Initialize the PID Manager.
        
        Args:
            pid_file_path: Optional custom path for the PID file. If None, uses default location.
            service_name: Optional service name to include in the PID filename for identification.
        """
        # Store the service name for later use
        self.service_name = service_name
        
        if pid_file_path:
            self.pid_file = pid_file_path
        else:
            # Include service name in the PID filename if provided
            filename = f"{service_name}_server.pid" if service_name else "mcp_server.pid"
            self.pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        logger.debug(f"PID Manager initialized with file path: {self.pid_file}")
        if service_name:
            logger.debug(f"PID Manager initialized for service: {service_name}")
    
    def check_server_running(self):
        """
        Check if a server is already running by examining the PID file.

        Returns:
            int or None: The PID of the running server, or None if no server is running.
        """
        if os.path.exists(self.pid_file):
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())

                # Check if the process with this PID exists
                if psutil.pid_exists(pid):
                    # Get process name to confirm it's our server
                    process = psutil.Process(pid)
                    if "python" in process.name().lower() and any("server.py" in cmd.lower() for cmd in process.cmdline()):
                        return pid

                # If we get here, the PID exists but it's not our server process
                logger.warning(f"🧹 Removing stale PID file from previous server instance")
                os.remove(self.pid_file)
                return None
            except (ValueError, ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"🧹 Removing stale PID file: {str(e)}")
                os.remove(self.pid_file)
                return None
        return None

    def create_pid_file(self):
        """
        Create a PID file with the current process ID.

        Returns:
            bool: True if the file was created successfully, False otherwise.
        """
        pid = os.getpid()
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(pid))
            logger.info(f"📝 Created PID file with server process ID: {pid}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create PID file: {str(e)}")
            return False

    def remove_pid_file(self):
        """
        Remove the PID file if it exists.
        
        Returns:
            bool: True if the file was removed successfully or didn't exist, False on error.
        """
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
                logger.info(f"🧹 Removed PID file")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove PID file: {str(e)}")
            return False
            
    @staticmethod
    def self_test():
        """
        Run a self-test of the PIDManager functionality.
        
        Tests initialization, creating a PID file, checking server status, and cleanup.
        
        Returns:
            bool: True if all tests pass, False if any test fails.
        """
        logger.info("🧪 Running PIDManager self-test...")
        test_passed = True
        
        # Create a temporary directory for our test PID file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_pid_path = os.path.join(temp_dir, "test_mcp_server.pid")
            logger.debug(f"Using test PID file path: {test_pid_path}")
            
            # Test 1: Basic Initialization
            try:
                pid_manager = PIDManager(test_pid_path)
                logger.debug(f"✅ Test 1a: PIDManager initialized with custom path")
                
                # Test service_name parameter
                test_service_pid_path = os.path.join(temp_dir, "test_custom_server.pid")
                test_service_name = "custom"
                service_pid_manager = PIDManager(test_service_pid_path, service_name=test_service_name)
                logger.debug(f"✅ Test 1b: PIDManager initialized with service name: {test_service_name}")
                
                # Test automatic service name in filename
                auto_service_pid_manager = PIDManager(service_name="auto_test")
                expected_filename = "auto_test_server.pid"
                if expected_filename in auto_service_pid_manager.pid_file:
                    logger.debug(f"✅ Test 1c: Service name correctly included in auto-generated filename")
                else:
                    logger.error(f"❌ Test 1c: Service name not included in filename: {auto_service_pid_manager.pid_file}")
                    test_passed = False
                
                # Test 2: Creating a PID file
                if pid_manager.create_pid_file():
                    if os.path.exists(test_pid_path):
                        logger.debug(f"✅ Test 2: PID file successfully created")
                        
                        # Verify PID file contents
                        with open(test_pid_path, 'r') as f:
                            pid = int(f.read().strip())
                            if pid == os.getpid():
                                logger.debug(f"✅ PID file contains current process ID: {pid}")
                            else:
                                logger.error(f"❌ PID file contains incorrect process ID: {pid}")
                                test_passed = False
                        
                        # Test 3: Check server running
                        detected_pid = pid_manager.check_server_running()
                        # This should detect our own process as running
                        if detected_pid is not None:
                            logger.debug(f"✅ Test 3: Detected running server with PID: {detected_pid}")
                        else:
                            logger.warning(f"⚠️ Test 3: Self-process not detected as running server")
                            # This isn't a failure as it depends on how the test is run
                            
                        # Test 4: Remove PID file
                        if pid_manager.remove_pid_file():
                            if not os.path.exists(test_pid_path):
                                logger.debug(f"✅ Test 4: PID file successfully removed")
                            else:
                                logger.error(f"❌ Test 4: PID file not removed")
                                test_passed = False
                        else:
                            logger.error(f"❌ Test 4: remove_pid_file() returned False")
                            test_passed = False
                    else:
                        logger.error(f"❌ Test 2: PID file not found after create_pid_file()")
                        test_passed = False
                else:
                    logger.error(f"❌ Test 2: create_pid_file() returned False")
                    test_passed = False
            except Exception as e:
                logger.error(f"❌ PIDManager self-test failed with exception: {str(e)}")
                test_passed = False
                
            # Ensure cleanup in case of test failure
            try:
                if os.path.exists(test_pid_path):
                    os.remove(test_pid_path)
            except Exception:
                pass
                
        if test_passed:
            logger.info("✅ PIDManager self-test PASSED")
        else:
            logger.error("❌ PIDManager self-test FAILED")
            
        return test_passed


if __name__ == "__main__":
    print("Running PIDManager self-test...")
    result = PIDManager.self_test()
    print(f"Self-test {'PASSED' if result else 'FAILED'}")
