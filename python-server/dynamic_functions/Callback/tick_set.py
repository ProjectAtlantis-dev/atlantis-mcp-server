import atlantis
import json
import logging
import os

logger = logging.getLogger("mcp_server")

_TICK_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "Data", "tick.json"
)


def _read_tick_config() -> dict:
    try:
        with open(_TICK_CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"tick config read failed: {e}")
        return {}


def _write_tick_config(cfg: dict):
    os.makedirs(os.path.dirname(_TICK_CONFIG_PATH), exist_ok=True)
    with open(_TICK_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_tick_tool_path() -> str:
    """Return the configured tick toolPath, or the default 'Callback.tick_callback'."""
    return _read_tick_config().get("toolPath") or "Callback.tick_callback"


def get_tick_enabled() -> bool:
    """Return whether ticking is explicitly enabled."""
    return bool(_read_tick_config().get("enabled", False))


def set_tick_enabled(enabled: bool):
    """Persist whether ticking is explicitly enabled."""
    cfg = _read_tick_config()
    cfg["enabled"] = bool(enabled)
    _write_tick_config(cfg)


@visible
async def tick_set(toolPath: str) -> str:
    """Set the global tick function by toolPath (e.g. 'Callback.tick_callback').

    The toolPath is a dotted path under dynamic_functions/, where the last
    segment is both the module name and the function name to invoke each tick.
    The value is persisted to Data/tick.json so it survives reloads.
    """
    if not toolPath or not isinstance(toolPath, str):
        raise ValueError("tick_set requires a non-empty toolPath string")

    cfg = _read_tick_config()
    cfg["toolPath"] = toolPath
    _write_tick_config(cfg)

    logger.info(f"tick_set: tick toolPath set to {toolPath}")
    return f"tick toolPath set to {toolPath}"
