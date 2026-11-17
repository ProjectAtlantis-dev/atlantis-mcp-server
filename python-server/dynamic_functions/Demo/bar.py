import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@visible
async def bar() -> dict[str, object]:
    """Return caller context info"""
    return {
        "user_game_id": atlantis.get_user_game_id(),
        "caller_sid": atlantis.get_caller(),
        "caller_shell": atlantis.get_caller_shell_path(),
        "exec_shell": atlantis.get_exec_shell_path(),
        "display_shell": atlantis.get_display_shell_path(),
        "request_id": atlantis.get_request_id(),
        "session_key": atlantis.get_session_key(),
        "entry_point": atlantis.get_entry_point_name(),
    }
