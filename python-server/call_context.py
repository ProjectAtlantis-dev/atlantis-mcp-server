from typing import Optional
from pydantic import BaseModel, ConfigDict


class CallContext(BaseModel):
    """Per-request context for a tool call.

    Built once at the JSON-RPC boundary in server._handle_tools_call,
    then passed by reference through the execution chain. New fields
    land here — not on intermediate function signatures.
    """
    model_config = ConfigDict(frozen=True, extra="ignore", coerce_numbers_to_str=True)

    client_id: Optional[str] = None
    request_id: Optional[str] = None
    user: Optional[str] = None
    session_id: Optional[str] = None
    command_seq: Optional[int] = None
    shell_path: Optional[str] = None

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
            user=params.get("user"),
            session_id=params.get("session_id"),
            command_seq=params.get("command_seq"),
            shell_path=params.get("shell_path"),
        )
