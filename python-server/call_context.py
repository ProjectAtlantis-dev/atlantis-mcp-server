from typing import Any, Callable, Optional
from pydantic import BaseModel, ConfigDict


class CallContext(BaseModel):
    """Per-request context for a tool call.

    Built once at the JSON-RPC boundary in server._handle_tools_call,
    then passed by reference through the execution chain. New fields
    land here — not on intermediate function signatures.

    The cloud currently sends params.user as the caller sid. Convert that
    wire-format name to caller_sid at this boundary; ownership is separate.
    """
    model_config = ConfigDict(frozen=True, extra="ignore", coerce_numbers_to_str=True)

    client_id: Optional[str] = None
    request_id: Optional[str] = None
    caller_sid: Optional[str] = None
    user_game_id: Optional[int] = None
    exec_shell_path: Optional[str] = None
    caller_shell_path: Optional[str] = None
    client_log_func: Optional[Callable[..., Any]] = None
    entry_point_name: Optional[str] = None

    @property
    def user(self) -> Optional[str]:
        """Backward-compatible alias for caller_sid."""
        return self.caller_sid

    @property
    def session_key(self) -> str:
        """Stable, locally-derived session identifier.

        Composed from (user_game_id, caller sid, caller_shell_path). Raises
        if any component is missing — a partial session_key is meaningless.
        """
        return self.derive_session_key(
            user_game_id=self.user_game_id,
            caller_sid=self.caller_sid,
            caller_shell_path=self.caller_shell_path,
        )

    def get_session_key(self) -> Optional[str]:
        """Return the canonical session key, or None when context is incomplete."""
        try:
            return self.session_key
        except ValueError:
            return None

    @staticmethod
    def derive_session_key(
        *,
        user_game_id: Optional[int],
        caller_sid: Optional[str],
        caller_shell_path: Optional[str],
    ) -> str:
        """Canonical session key factory for cloud tool-call context."""
        missing = [
            n for n, v in (
                ("user_game_id", user_game_id),
                ("caller_sid", caller_sid),
                ("caller_shell_path", caller_shell_path),
            ) if not v
        ]
        if missing:
            raise ValueError(f"Cannot derive session_key: missing {missing}")
        return f"{user_game_id}:{caller_sid}:{caller_shell_path}"

    @classmethod
    def from_params(
        cls,
        params: dict,
        client_id: Optional[str],
        request_id: Optional[str],
    ) -> "CallContext":
        return cls(
            client_id=client_id,
            request_id=request_id,
            caller_sid=params.get("caller_sid") or params.get("user"),
            user_game_id=params.get("user_game_id"),
            exec_shell_path=params.get("exec_shell_path"),
            caller_shell_path=params.get("caller_shell_path"),
        )
