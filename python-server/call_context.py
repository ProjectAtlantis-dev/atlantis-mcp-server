from typing import Any, Callable, Optional
from pydantic import BaseModel, ConfigDict, Field


class ToolCallPayload(BaseModel):
    """Typed JSON-RPC tools/call params payload carried by CallContext."""
    model_config = ConfigDict(frozen=True, extra="allow", coerce_numbers_to_str=True)

    name: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    caller_sid: Optional[str] = None
    user: Optional[str] = None
    user_game_id: Optional[int] = None
    exec_shell_path: Optional[str] = None
    caller_shell_path: Optional[str] = None


class CallContext(BaseModel):
    """Per-request context for a tool call.

    Built once at the JSON-RPC boundary in server._handle_tools_call,
    then passed by reference through the execution chain. New fields
    land here — not on intermediate function signatures.

    The wire payload is kept intact on payload. Compatibility accessors expose
    the context fields callers use without copying them into parallel state.
    """
    model_config = ConfigDict(frozen=True, extra="ignore", coerce_numbers_to_str=True)

    client_id: Optional[str] = None
    request_id: Optional[str] = None
    payload: ToolCallPayload
    # exec_shell_path: shell where this tool call's work runs. Outbound
    # client_command callbacks must be tagged with THIS so they nest under
    # the tool call in the cloud's command tree.
    # caller_shell_path: user's root shell - who triggered the call.
    # Attribution / session routing only; never use for placing work.
    # These are distinct on purpose - don't swap them.
    client_log_func: Optional[Callable[..., Any]] = None
    entry_point_name: Optional[str] = None

    @property
    def caller_sid(self) -> Optional[str]:
        """Caller sid. Cloud legacy wire format may send this as params.user."""
        return self.payload.caller_sid or self.payload.user

    @property
    def user(self) -> Optional[str]:
        """Backward-compatible alias for caller_sid."""
        return self.caller_sid

    @property
    def user_game_id(self) -> Optional[int]:
        return self.payload.user_game_id

    @property
    def exec_shell_path(self) -> Optional[str]:
        return self.payload.exec_shell_path

    @property
    def caller_shell_path(self) -> Optional[str]:
        return self.payload.caller_shell_path

    @property
    def session_key(self) -> str:
        """Stable, locally-derived session identifier.

        Composed from (user_game_id, caller sid). Shell path is intentionally NOT included —
        we want one session per (game, sid) so multiple terminals of the same human share state
        (e.g. the chat busy-lock). Raises if any component is missing.
        """
        return self.derive_session_key(
            user_game_id=self.user_game_id,
            caller_sid=self.caller_sid,
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
    ) -> str:
        """Canonical session key factory for cloud tool-call context."""
        missing = [
            n for n, v in (
                ("user_game_id", user_game_id),
                ("caller_sid", caller_sid),
            ) if not v
        ]
        if missing:
            raise ValueError(f"Cannot derive session_key: missing {missing}")
        return f"{user_game_id}:{caller_sid}"

    def with_payload_updates(self, **updates: Any) -> "CallContext":
        """Return a copy with selected wire payload keys overridden."""
        payload = ToolCallPayload.model_validate({**self.payload.model_dump(exclude_unset=True), **updates})
        return self.model_copy(update={"payload": payload})

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
            payload=ToolCallPayload.model_validate(params),
        )
