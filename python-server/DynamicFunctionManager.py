class DynamicFunctionManager:
    def __init__(self, functions_dir):
        # State that was previously global
        self.functions_dir = functions_dir
        self._runtime_errors = {}
        self._dynamic_functions_cache = {}
        self._dynamic_load_lock = asyncio.Lock()

        # Create directories if they don't exist
        os.makedirs(self.functions_dir, exist_ok=True)
        self.old_dir = os.path.join(self.functions_dir, "OLD")
        os.makedirs(self.old_dir, exist_ok=True)

    # File operations
    async def _fs_save_code(self, name, code):
        # Implementation...

    async def _fs_load_code(self, name):
        # Implementation...

    # Metadata extraction and validation
    def _code_extract_basic_metadata(self, code_buffer):
        # Implementation...

    def _code_validate_syntax(self, code_buffer):
        # Implementation...

    # Cache management
    async def invalidate_all_dynamic_module_cache(self):
        # Implementation...

    # Public API methods
    async def function_add(self, name, code=None):
        # Implementation...

    async def function_remove(self, name):
        # Implementation...

    async def function_set(self, code_buffer, server):
        # Implementation...

    async def function_validate(self, name):
        # Implementation...

    async def function_call(self, name, client_id, request_id, **kwargs):
        # Implementation...

    async def get_function_code(self, name):
        # Implementation...