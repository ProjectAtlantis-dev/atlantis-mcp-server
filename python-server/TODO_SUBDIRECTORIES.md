# TODO: Dynamic Functions Subdirectory Support

## Overview
Implement subdirectory support for dynamic functions where all functions must be organized in subfolders, with a default "Home" folder. The folder name serves as an alternative way to specify the app name, complementing the existing `@app` decorator.

## Directory Structure (One Level Deep)
```
dynamic_functions/
├── Home/                    # Default folder for general functions
│   ├── chat.py
│   ├── lobby.py
│   └── utilities.py
├── chat_app/               # One level deep - app_name = "chat_app"
│   ├── message_handlers.py
│   ├── user_management.py
│   └── notifications.py
├── admin_app/              # One level deep - app_name = "admin_app"
│   ├── user_management.py
│   ├── system_monitoring.py
│   └── configuration.py
└── utilities/              # One level deep - app_name = "utilities"
    ├── database.py
    └── logging.py
```

**App Name Resolution**: Always use the first (and only) subdirectory name as the app name.
**Module Names**: `dynamic_functions.app_name.filename` (e.g., `dynamic_functions.chat_app.message_handlers`)

**App/Folder Name Validation**:
- Allow mixed case (e.g., `ChatApp`, `chat_app`, `CHAT_APP`)
- Allow dashes and underscores
- Throw exception for spaces or invalid characters
- No auto-cleaning or sanitization

**Exception Handling Principles**:
- **NO exception catching/burying** unless there is an exceptional reason
- Let exceptions bubble up naturally to the caller
- Clear error messages in exceptions
- Fail fast and fail loudly

## Implementation Phases

### Phase 1: Core File Discovery (1 hour)
**Goal**: Make the system scan subdirectories without changing file operations

#### Changes Required:
- [ ] Update `_build_function_file_mapping()` in DynamicFunctionManager.py
  - [ ] Change from `os.listdir()` to `os.walk()`
  - [ ] Skip root directory in scanning
  - [ ] Store relative paths in function mapping
- [ ] Update tool discovery in server.py `_get_tools_list()`
  - [ ] Change from flat file scanning to recursive scanning
  - [ ] Handle relative paths in function file list

#### Testing:
- [ ] Create test subdirectories manually
- [ ] Verify functions are discovered correctly
- [ ] Ensure existing flat structure still works (if any files exist)
- [ ] Test function-to-file mapping cache invalidation

#### Rollback: Easy - just revert the scanning logic

---

### Phase 2: File Operations with Default "Home" (1 hour)
**Goal**: Make file save/load work with subdirectories

#### Changes Required:
- [ ] Update `_fs_save_code()` in DynamicFunctionManager.py
  - [ ] Add `app_name` parameter with default "Home"
  - [ ] Validate app_name format (no spaces, valid characters only)
  - [ ] Create subdirectory if it doesn't exist
  - [ ] Save files to appropriate subdirectory
- [ ] Update `_fs_load_code()` in DynamicFunctionManager.py
  - [ ] Add `app_name` parameter with default "Home"
  - [ ] Load files from subdirectories
- [ ] Update `function_add()` in DynamicFunctionManager.py
  - [ ] Create new functions in "Home" subdirectory by default
  - [ ] Ensure subdirectory exists before creating file

#### Testing:
- [ ] Test `_function_add` tool creates files in Home/
- [ ] Test `_function_set` tool saves to correct subdirectory
- [ ] Test `_function_get` tool loads from subdirectories
- [ ] Test file operations with custom app names
- [ ] Test validation of app names in file operations

#### Rollback: Easy - revert file operation changes

---

### Phase 3: Module Loading (1 hour)
**Goal**: Make Python imports work with subdirectory paths

#### Changes Required:
- [ ] Update `function_call()` in DynamicFunctionManager.py
  - [ ] Update module name generation for one-level subdirectories
  - [ ] Convert file paths to proper module names (e.g., `Home.chat` instead of `chat`)
- [ ] Update parent package path handling
  - [ ] Ensure `__path__` includes subdirectories
  - [ ] Handle module imports correctly

#### Testing:
- [ ] Test function execution from subdirectories
- [ ] Verify module imports work correctly
- [ ] Test function calling with new path structure
- [ ] Test module cache invalidation

#### Rollback: Easy - revert module loading changes

---

### Phase 4: App Name Resolution (1 hour)
**Goal**: Extract app names from folder paths

#### Changes Required:
- [ ] Add `_get_app_name_from_path()` method in DynamicFunctionManager.py
  - [ ] Extract app name from first subdirectory only
  - [ ] Validate app name format (no spaces, valid characters only)
  - [ ] Handle edge cases (files in root directory - throw exception)
- [ ] Update `_code_validate_syntax()` in DynamicFunctionManager.py
  - [ ] Integrate folder-based app name detection
  - [ ] Handle conflict resolution between @app decorator and folder name
  - [ ] Update function info to include folder-derived app names
- [ ] Update tool annotations in server.py
  - [ ] Include folder-derived app names in tool annotations

#### Testing:
- [ ] Test app name extraction from first-level subdirectories
- [ ] Test conflict resolution between @app decorator and folder name
- [ ] Verify tool annotations show correct app names
- [ ] Test with various one-level folder structures
- [ ] Test validation of app names (reject spaces, invalid chars)
- [ ] Test mixed case app names are accepted

#### Rollback: Easy - revert app name logic

---

### Phase 5: File Watcher (30 minutes)
**Goal**: Make file watcher monitor subdirectories

#### Changes Required:
- [ ] Update file watcher setup in server.py
  - [ ] Change `recursive=False` to `recursive=True`
  - [ ] Test file watcher event handling for subdirectories

#### Testing:
- [ ] Test file changes in subdirectories trigger reloads
- [ ] Verify cache invalidation works correctly
- [ ] Test file creation/deletion in subdirectories

#### Rollback: Easy - revert to `recursive=False`

---

## Testing Strategy

### Manual Testing Checklist for Each Phase:

#### Phase 1 Testing:
```bash
# Create test structure
mkdir -p dynamic_functions/Home
mkdir -p dynamic_functions/chat_app
echo "async def test_func(): return 'test'" > dynamic_functions/Home/test_function.py
echo "async def message_handler(): return 'message'" > dynamic_functions/chat_app/message_handler.py

# Verify discovery
# Check that both functions are found in tool list
```

#### Phase 2 Testing:
```bash
# Test file operations
# Use _function_add tool - should create in Home/
# Use _function_set tool with different app names
# Use _function_get tool to retrieve code
```

#### Phase 3 Testing:
```bash
# Test function execution
# Call functions from different subdirectories
# Verify imports work correctly
```

#### Phase 4 Testing:
```bash
# Test app name resolution
# Check tool annotations show correct app names
# Test @app decorator vs folder name conflicts
```

#### Phase 5 Testing:
```bash
# Test file watcher
# Modify files in subdirectories
# Verify automatic reloads
```

### Automated Testing (Future Enhancement):
- [ ] Create unit tests for each phase
- [ ] Add integration tests for full workflow
- [ ] Add regression tests for existing functionality

## Rollback Strategy

### If Phase 1 Fails:
- Revert scanning changes
- System continues to work with flat structure

### If Phase 2 Fails:
- Revert file operation changes
- System can still discover files but can't create/load them

### If Phase 3 Fails:
- Revert module loading changes
- System can discover and save files but can't execute them

### If Phase 4 Fails:
- Revert app name resolution
- System works but without folder-based app names

### If Phase 5 Fails:
- Revert file watcher changes
- System works but requires manual cache clearing

## Files to Modify

### Primary Changes:
- [ ] `DynamicFunctionManager.py` - Core file management logic
- [ ] `server.py` - Tool discovery and file watcher

### No Changes Needed:
- [ ] `state.py` - Configuration remains the same
- [ ] `atlantis.py` - Context handling unchanged
- [ ] `DynamicServerManager.py` - Unrelated to function management
- [ ] `ColoredFormatter.py` - Utility unchanged
- [ ] `utils.py` - No changes needed

## Success Criteria

### Phase 1 Success:
- [ ] Functions in subdirectories are discovered
- [ ] Function-to-file mapping works correctly
- [ ] No regressions in existing functionality

### Phase 2 Success:
- [ ] Files can be saved to subdirectories
- [ ] Files can be loaded from subdirectories
- [ ] Default "Home" folder is used when no app specified

### Phase 3 Success:
- [ ] Functions can be executed from subdirectories
- [ ] Module imports work correctly
- [ ] No import errors or module loading issues

### Phase 4 Success:
- [ ] App names are correctly extracted from folder paths
- [ ] Tool annotations show correct app names
- [ ] @app decorator takes precedence over folder names

### Phase 5 Success:
- [ ] File changes in subdirectories trigger reloads
- [ ] Cache invalidation works correctly
- [ ] No performance issues with recursive watching

## Total Estimated Effort: 4.5 hours

## Notes
- Assume dynamic_functions directory is empty (no migration needed)
- **One level of depth only** - simplifies app naming and implementation
- **No auto-cleaning of app names** - throw exceptions for invalid characters
- **No exception catching/burying** - let exceptions bubble up unless there is an exceptional reason
- Each phase builds on the previous one
- Easy rollback at each phase
- Comprehensive testing at each step
- Clear success criteria for each phase