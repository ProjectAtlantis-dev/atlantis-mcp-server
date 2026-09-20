"""Explicit owner-only enrollment using the MCP host's authenticated owner context.

No guessed cloud database IDs and no account selection in player commands.
Existing policies are preserved, not automatically migrated or overwritten.
"""
import json
import os
from pathlib import Path
import re
from uuid import uuid4


def enroll_owner(world):
    import atlantis
    from atlantis_host_adapters.identity import current_principal
    from .gateway import account_for
    context = atlantis.get_context()
    if (context is None or context.caller_sid not in atlantis.get_owner_usernames()
            or type(context.user_game_id) is not int or context.user_game_id <= 0):
        raise PermissionError("Authenticated MCP host owner context required")
    if not isinstance(world, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", world):
        raise ValueError("Invalid persistent world name")
    configured = os.environ.get("ATLANTIS_IDENTITY_BINDINGS")
    if not configured or not Path(configured).is_absolute():
        raise RuntimeError("Configure an absolute ATLANTIS_IDENTITY_BINDINGS path")
    path = Path(configured)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: concurrent calls must never replace an existing subject.
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        policy = {"version": 2, "bindings": [{"caller": context.caller_sid,
                  "subject_uuid": str(uuid4()), "scenario": world,
                  "permissions": ["bank", "simulation", "world"]}]}
        with os.fdopen(descriptor, "w") as stream:
            json.dump(policy, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    principal = current_principal("bank")
    if principal.scenario != world:
        raise ValueError("Existing owner binding selects another world; it was not changed")
    account = account_for(principal)
    return {"world": world, "account": account, "subject": principal.external_user_id,
            "visibility": "owner-only", "policyPath": str(path)}
