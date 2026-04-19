"""Bot-owned chat dispatch and default chat-completions handler."""

import inspect
import os
import time as _t
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from dynamic_functions.Home.bot_common import (
    TranscriptToolT,
    ToolLookupInfo,
    get_base_tools,
    logger,
)
from dynamic_functions.Home.turn import run_turn


ProcedureInjectionProvider = Callable[["BotChatContext"], Any]
ChatHandler = Callable[["BotChatContext"], Any]


@dataclass
class BotChatContext:
    """Everything a bot runtime needs to decide how to handle a chat turn."""

    session_id: str
    request_id: str
    game_id: Optional[str]
    caller: str
    bot_sid: str
    bot_cfg: Dict[str, Any]
    role: Dict[str, Any]
    user_profile: Optional[Dict[str, Any]]
    interaction: Dict[str, Any]
    raw_transcript: List[Dict[str, Any]]
    transcript: List[Dict[str, Any]]
    procedure_injection_provider: Optional[ProcedureInjectionProvider] = None
    skip_post_turn_record: bool = False

    @property
    def location(self) -> str:
        return str(self.role.get("location", ""))

    @property
    def role_title(self) -> str:
        return str(self.role.get("title", "Assistant"))

    @property
    def bot_display_name(self) -> str:
        return str(self.bot_cfg.get("displayName", self.bot_sid))

    @property
    def guest(self) -> Optional[Dict[str, Any]]:
        """Scenario compatibility for check-in modules that still say guest."""
        return self.user_profile

    async def build_procedure_injections(self) -> List[Dict[str, Any]]:
        if not self.procedure_injection_provider:
            return []

        result = self.procedure_injection_provider(self)
        if inspect.isawaitable(result):
            result = await result
        return list(result or [])


async def dispatch_chat(context: BotChatContext) -> Optional[str]:
    """Route chat handling to the bot's configured handler, or the default runtime."""
    handler_path = context.bot_cfg.get("chatHandler")
    handler = _load_chat_handler(handler_path) if handler_path else handle_chat_completions

    result = handler(context)
    if inspect.isawaitable(result):
        return await result
    return result


async def handle_chat_completions(context: BotChatContext) -> Optional[str]:
    """Default bot runtime for streaming chat-completions providers."""
    bot_cfg = context.bot_cfg
    bot_sid = str(bot_cfg["sid"])
    bot_display_name = context.bot_display_name

    if _last_chat_sid(context.raw_transcript) == bot_sid:
        logger.warning(f"Last chat was from {bot_display_name} - skipping")
        context.skip_post_turn_record = True
        return None

    t0 = _t.monotonic()
    system_prompt = await _build_system_prompt(context)
    logger.info(f"System prompt built in {_t.monotonic() - t0:.2f}s ({len(system_prompt)} chars)")

    injections = await context.build_procedure_injections()
    for injection in injections:
        context.transcript.append(injection)
    if injections:
        logger.info(f"Injected {len(injections)} procedure message(s) from role {context.role.get('id')}")

    converted_tools, tool_lookup = await _build_tools(context)
    client = _build_chat_completions_client(bot_cfg)

    allowed_apps = context.role.get("allowedApps")
    logger.info(
        f"=== HANDING OFF TO {bot_display_name} "
        f"({context.role_title} at {context.location}) === "
        f"session={context.session_id} allowed_apps={allowed_apps}"
    )

    return await run_turn(
        client=client,
        model=str(bot_cfg["model"]),
        bot_sid=bot_sid,
        bot_display_name=bot_display_name,
        system_prompt=system_prompt,
        transcript=context.transcript,
        converted_tools=converted_tools,
        tool_lookup=tool_lookup,
        sessionId=context.session_id,
        requestId=context.request_id,
        allowed_apps=allowed_apps,
    )


def _load_chat_handler(handler_path: str) -> ChatHandler:
    module_name, attr_name = _split_dotted_callable(handler_path)
    handler = getattr(import_module(module_name), attr_name)
    if not callable(handler):
        raise TypeError(f"Configured chatHandler is not callable: {handler_path}")
    return handler


def _split_dotted_callable(path: str) -> tuple[str, str]:
    normalized = path.replace(":", ".")
    if "." not in normalized:
        raise ValueError(f"chatHandler must be a dotted callable path: {path}")
    return normalized.rsplit(".", 1)


def _last_chat_sid(raw_transcript: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(raw_transcript):
        if msg.get("type") == "chat" and msg.get("sid") != "system":
            return msg.get("sid")
    return None


async def _build_system_prompt(context: BotChatContext) -> str:
    bot_cfg = context.bot_cfg
    bot_dir = bot_cfg.get("_botDir", "")

    # Read base prompt from markdown file
    prompt_file = bot_cfg.get("systemPrompt", "system_prompt.md")
    prompt_path = os.path.join(bot_dir, prompt_file) if bot_dir else prompt_file
    if not os.path.isfile(prompt_path):
        raise ValueError(f"System prompt not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read().strip()
    if not base_prompt:
        raise ValueError(f"System prompt is empty: {prompt_path}")

    # Apply prompt builder (adds datetime, interaction context, etc.)
    prompt_builder_path = bot_cfg.get("promptBuilder")
    if not prompt_builder_path:
        # Default: look for prompt.py next to system_prompt.md
        prompt_py = os.path.join(bot_dir, "prompt.py") if bot_dir else None
        if prompt_py and os.path.isfile(prompt_py):
            # Derive module path from bot dir
            rel = os.path.relpath(prompt_py, os.path.join(bot_dir, "..", ".."))
            module_name = rel.replace(os.sep, ".").replace(".py", "")
            prompt_builder_path = f"dynamic_functions.{module_name}.build_system_prompt"

    if prompt_builder_path:
        module_name, attr_name = _split_dotted_callable(str(prompt_builder_path))
        build_system_prompt = getattr(import_module(module_name), attr_name)
        prompt_caller, first_name, interaction_count, last_interaction = _interaction_prompt_context(context)
        system_prompt = build_system_prompt(
            base_prompt,
            prompt_caller,
            interaction_count,
            last_interaction,
            first_name=first_name,
        )
        if inspect.isawaitable(system_prompt):
            system_prompt = await system_prompt
        return str(system_prompt)

    return base_prompt


def _interaction_prompt_context(context: BotChatContext) -> tuple[str, str, int, str]:
    interaction_count = int(context.interaction.get("prior_interaction_count") or 0)
    last_interaction = str(context.interaction.get("last_interaction_at") or "")
    first_name = str(context.user_profile.get("first_name", "")) if context.user_profile else ""
    return context.caller, first_name, interaction_count, last_interaction


async def _build_tools(context: BotChatContext) -> tuple[List[TranscriptToolT], Dict[str, ToolLookupInfo]]:
    tool_builder_path = context.bot_cfg.get("toolBuilder")
    if not tool_builder_path:
        return get_base_tools()

    module_name, attr_name = _split_dotted_callable(str(tool_builder_path))
    builder = getattr(import_module(module_name), attr_name)
    result = builder(context)
    if inspect.isawaitable(result):
        result = await result
    return result


def _build_chat_completions_client(bot_cfg: Dict[str, Any]) -> OpenAI:
    api_key_env = str(bot_cfg["apiKeyEnv"])
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} environment variable is not set")

    return OpenAI(
        api_key=api_key,
        base_url=bot_cfg["baseUrl"],
    )
