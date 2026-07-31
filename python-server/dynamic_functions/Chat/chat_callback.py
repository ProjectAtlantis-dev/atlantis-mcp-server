"""Game chat callback — main game tick. Fired on every transcript change."""

import atlantis
import copy
import logging
import os
import re
from typing import Any, Dict, List, Optional


from .chat import (
    analyze_participants, fetch_transcript,
)
from .bot_tool import get_bot_tool_argument_overrides, get_bot_tools
from .common import _read_json, _write_json
from .game import _game_is_running, _game_roster_scene, game_find_current, require_membership
from .roster import _load_game_roster
from .turn import bot_turn

logger = logging.getLogger("dynamic_function")

_BUSY_KEY = "chat_busy"
_LAST_CHAT_KEY_PREFIX = "chat_last_seen:"
_CHAT_LOOP_COUNT_PREFIX = "chat_loop_count:"
_IDENTITY_KNOWLEDGE_FILENAME = "identity_knowledge.json"
_UNKNOWN_VISITOR_NAME = "a visitor"
_MAX_BOT_CHAIN = 4


def _require_roster_assigned(game_key: str) -> None:
    """Fail early if chat starts before this game has a created roster."""
    data_dir = require_membership(game_key)
    meta = _read_json(os.path.join(data_dir, "game.json")) or {}
    if not _game_roster_scene(meta):
        raise RuntimeError(f"Game {game_key!r} has no roster assigned yet")
    if not os.path.isfile(os.path.join(data_dir, "roster.json")):
        raise RuntimeError(f"Game {game_key!r} has no roster.json yet")

@public
@preflight
async def preflight_callback():
    await atlantis.client_log("doing preflight")





@public
@chat
async def chat_callback():
    """Game tick: attach this chat session to a game, then respond to transcript changes."""
    if not atlantis.get_session_key():
        logger.warning("chat_callback fired without session context, skipping")
        return

    request_id = atlantis.get_request_id() or "unknown"
    logger.info(
        "chat_callback start: request_id=%s session=%s caller=%s user_game_id=%s",
        request_id,
        atlantis.get_session_key(),
        atlantis.get_caller(),
        atlantis.get_user_game_id(),
    )
    if atlantis.session_shared.get(_BUSY_KEY):
        logger.info("chat_callback busy, skipping request_id=%s", request_id)
        return

    atlantis.session_shared.set(_BUSY_KEY, request_id)
    try:
        game_key = await game_find_current()
        logger.info("chat_callback game resolved: %s", game_key)
        if not _game_is_running(game_key):
            logger.info("chat_callback game %r is stopped, skipping", game_key)
            await atlantis.client_log(f"chat_callback skipped: game {game_key!r} is stopped")
            return
        _require_roster_assigned(game_key)
        await _handle_chat(game_key)
    finally:
        atlantis.session_shared.remove(_BUSY_KEY)


async def _handle_chat(game_key: str):
    logger.info("chat_callback handle_chat: game=%s", game_key)
    raw_transcript, transcript = await fetch_transcript(game_key)
    participants = analyze_participants(raw_transcript)
    speaker_sid = participants.get("last_speaker")
    logger.info(
        "chat_callback transcript: game=%s raw=%s filtered=%s last_speaker=%r participants=%s",
        game_key,
        len(raw_transcript),
        len(transcript),
        speaker_sid,
        sorted((participants.get("participants") or {}).keys()),
    )
    if not speaker_sid:
        await atlantis.client_log("No chat speaker found in transcript")
        return

    signature = _last_chat_signature(raw_transcript)
    if signature:
        last_key = f"{_LAST_CHAT_KEY_PREFIX}{game_key}"
        if atlantis.session_shared.get(last_key) == signature:
            logger.info("chat_callback duplicate transcript trigger, skipping")
            return
        atlantis.session_shared.set(last_key, signature)

    roster = _load_game_roster(game_key)
    speaker = _find_roster_speaker(roster, speaker_sid)
    if not speaker:
        await atlantis.client_log(
            f"Chat speaker {speaker_sid!r} is not in this game's roster. Check the roster before chatting."
        )
        return

    location = speaker.get("location")
    if not location:
        raise RuntimeError(f"Chat speaker {speaker_sid!r} has no current location yet")

    all_listeners = _all_listeners(roster, location)
    bot_listeners = _bot_listeners(roster, speaker)
    logger.info(
        "chat_callback listeners: game=%s speaker=%s location=%s all_listeners=%s bot_listeners=%s",
        game_key,
        _display_name(speaker),
        location,
        [_display_name(row) for row in all_listeners],
        [_display_name(row) for row in bot_listeners],
    )
    if not all_listeners:
        await atlantis.client_log(f"Room [{location}] is empty")
        return
    await atlantis.client_log(
        f"Room [{location}]: {', '.join(_display_name(row) for row in all_listeners)}"
    )
    if len(all_listeners) == 1:
        await atlantis.client_log(f"{_display_name(speaker)} is alone in {location}")
        return

    loop_count = _next_loop_count(game_key, speaker)
    if loop_count > _MAX_BOT_CHAIN:
        logger.debug("chat_callback bot chain limit reached, skipping")
        return

    if not bot_listeners:
        await atlantis.client_log(f"No AI roster member in {location} available to respond")
        return

    bot_record = bot_listeners[0]
    await atlantis.client_log(
        f"Next roster speaker: {bot_record.get('displayName', bot_record.get('bot_sid', 'bot'))}"
    )
    await _respond_as_bot(
        game_key=game_key,
        bot_record=bot_record,
        transcript=transcript,
        roster=roster,
    )


def _next_loop_count(game_key: str, speaker: Optional[Dict[str, Any]]) -> int:
    key = f"{_CHAT_LOOP_COUNT_PREFIX}{game_key}"
    if not speaker or not _is_ai(speaker):
        atlantis.session_shared.set(key, 0)
        return 0
    count = int(atlantis.session_shared.get(key) or 0) + 1
    atlantis.session_shared.set(key, count)
    return count


def _last_chat_signature(raw_transcript: List[Dict[str, Any]]) -> str:
    for msg in reversed(raw_transcript):
        if msg.get("type") != "chat":
            continue
        sid = str(msg.get("sid") or "")
        if not sid or sid == "system":
            continue
        if "thinking" in str(msg.get("who") or "").lower():
            continue
        content = str(msg.get("content") or "")
        if not content.strip():
            continue
        timestamp = str(msg.get("created_at") or msg.get("created_at_str") or "")
        return "|".join([sid, timestamp, content[:200]])
    return ""


def _is_ai(row: Dict[str, Any]) -> bool:
    return row.get("ai") is True


def _display_name(row: Dict[str, Any]) -> str:
    return row.get("displayName") or row.get("bot_sid") or row.get("sid") or row.get("key") or "unknown"


def _all_listeners(roster: List[Dict[str, Any]], location: str) -> List[Dict[str, Any]]:
    return [
        row for row in roster
        if row.get("location") == location
    ]


def _bot_listeners(roster: List[Dict[str, Any]], speaker: Dict[str, Any]) -> List[Dict[str, Any]]:
    """AI roster members in the same current location as the speaker."""
    location = speaker.get("location")
    if not location:
        return []
    return [
        row for row in _all_listeners(roster, location)
        if _is_ai(row) and row.get("bot_sid") and row.get("key") != speaker.get("key")
    ]


def _find_roster_speaker(roster: List[Dict[str, Any]], speaker_sid: str) -> Optional[Dict[str, Any]]:
    for row in roster:
        if not _is_ai(row) and row.get("sid") == speaker_sid:
            return row
    for row in roster:
        if _is_ai(row) and row.get("bot_sid") == speaker_sid:
            return row
    return None


def _identity_knowledge_path(game_key: str) -> str:
    return os.path.join(require_membership(game_key), _IDENTITY_KNOWLEDGE_FILENAME)


def _load_identity_knowledge(game_key: str) -> Dict[str, Dict[str, str]]:
    raw = _read_json(_identity_knowledge_path(game_key), {}) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Game {game_key!r} {_IDENTITY_KNOWLEDGE_FILENAME} must be a JSON object"
        )

    knowledge: Dict[str, Dict[str, str]] = {}
    for raw_bot_sid, raw_names in raw.items():
        if not isinstance(raw_names, dict):
            raise ValueError(
                f"Identity knowledge for bot {raw_bot_sid!r} must be an object"
            )
        names: Dict[str, str] = {}
        for raw_human_sid, raw_name in raw_names.items():
            human_sid = str(raw_human_sid or "").strip()
            name = str(raw_name or "").strip()
            if human_sid and name:
                names[human_sid] = name
        knowledge[str(raw_bot_sid)] = names
    return knowledge


def _known_human_names(game_key: str, bot_sid: str) -> Dict[str, str]:
    return dict(_load_identity_knowledge(game_key).get(bot_sid, {}))


def _mark_identity_known(
    game_key: str,
    *,
    bot_sid: str,
    human_sid: str,
    display_name: str,
) -> bool:
    bot_sid = str(bot_sid or "").strip()
    human_sid = str(human_sid or "").strip()
    display_name = str(display_name or "").strip()
    if not bot_sid or not human_sid or not display_name:
        return False

    knowledge = _load_identity_knowledge(game_key)
    known_by_bot = knowledge.setdefault(bot_sid, {})
    if known_by_bot.get(human_sid) == display_name:
        return False
    known_by_bot[human_sid] = display_name
    _write_json(_identity_knowledge_path(game_key), knowledge)
    logger.info(
        "Identity learned: game=%s bot=%s human=%s name=%r",
        game_key,
        bot_sid,
        human_sid,
        display_name,
    )
    return True


def _replace_name(value: str, display_name: str) -> str:
    return re.sub(
        rf"(?<!\w){re.escape(display_name)}(?!\w)",
        _UNKNOWN_VISITOR_NAME,
        value,
        flags=re.IGNORECASE,
    )


def _scrub_value(value: Any, unknown_names: List[str]) -> Any:
    if isinstance(value, str):
        scrubbed = value
        for display_name in sorted(unknown_names, key=len, reverse=True):
            scrubbed = _replace_name(scrubbed, display_name)
        return scrubbed
    if isinstance(value, list):
        return [_scrub_value(item, unknown_names) for item in value]
    if isinstance(value, dict):
        return {
            key: _scrub_value(item, unknown_names)
            for key, item in value.items()
        }
    return value


def _transcript_for_bot(
    *,
    game_key: str,
    bot_sid: str,
    transcript: List[Dict[str, Any]],
    roster: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    known_names = _known_human_names(game_key, bot_sid)
    unknown_names = [
        str(row.get("displayName") or "").strip()
        for row in roster
        if not _is_ai(row)
        and row.get("sid")
        and row.get("displayName")
        and known_names.get(str(row.get("sid"))) != str(row.get("displayName"))
    ]
    return _scrub_value(copy.deepcopy(transcript), unknown_names)


def _roster_names_for_bot(
    *,
    game_key: str,
    bot_sid: str,
    roster: List[Dict[str, Any]],
) -> Dict[str, str]:
    known_names = _known_human_names(game_key, bot_sid)
    names: Dict[str, str] = {}
    for row in roster:
        roster_bot_sid = str(row.get("bot_sid") or "").strip()
        if not roster_bot_sid:
            continue

        display_name = str(row.get("displayName") or "").strip()
        if _is_ai(row):
            if display_name:
                names[roster_bot_sid] = display_name
            continue

        human_sid = str(row.get("sid") or "").strip()
        known_name = known_names.get(human_sid) if human_sid else None
        names[roster_bot_sid] = known_name or _UNKNOWN_VISITOR_NAME
    return names


@public
async def remember_visitor(name: str, game_key: str, bot_sid: str) -> Dict[str, str]:
    """Remember the name a visitor directly told the listening bot.

    `game_key` is supplied by the Atlantis cursor. `bot_sid` is trusted turn
    context and is never exposed to the model.
    """
    name = str(name or "").strip()
    bot_sid = str(bot_sid or "").strip()
    if not name:
        raise ValueError("Visitor name is required")
    if not bot_sid:
        raise RuntimeError("No listening bot was supplied by the bot turn")

    roster = _load_game_roster(game_key)
    bot_record = next(
        (
            row for row in roster
            if _is_ai(row) and str(row.get("bot_sid") or "") == bot_sid
        ),
        None,
    )
    if not bot_record:
        raise ValueError(f"Bot {bot_sid!r} is not in game {game_key!r}")

    known_names = _known_human_names(game_key, bot_sid)
    visitors = [
        row for row in roster
        if not _is_ai(row)
        and row.get("sid")
        and row.get("location") == bot_record.get("location")
        and str(row.get("sid")) not in known_names
    ]
    if len(visitors) != 1:
        raise RuntimeError(
            f"Expected exactly one unknown visitor with {bot_sid!r}, found {len(visitors)}"
        )

    human_sid = str(visitors[0].get("sid") or "").strip()
    _mark_identity_known(
        game_key,
        bot_sid=bot_sid,
        human_sid=human_sid,
        display_name=name,
    )
    return {"visitor": name, "status": "remembered"}


async def greet_entrant(game_key: str, entrant_sid: str, location: str):
    """Fire an in-character greeting from a bot already at `location` toward a newcomer."""
    raise NotImplementedError("greet_entrant: slot system removed — needs reimplementation")


async def _respond_as_bot(
    *,
    game_key: str,
    bot_record: dict,
    transcript: list,
    roster: list,
):
    bot_sid = bot_record.get("bot_sid")
    if not bot_sid:
        raise ValueError(f"Roster row {bot_record.get('key')!r} has no bot_sid")

    bot_transcript = _transcript_for_bot(
        game_key=game_key,
        bot_sid=bot_sid,
        transcript=transcript,
        roster=roster,
    )
    roster_names = _roster_names_for_bot(
        game_key=game_key,
        bot_sid=bot_sid,
        roster=roster,
    )
    await atlantis.client_log(
        "respond_as_bot start: "
        f"game={game_key!r} slot={bot_record.get('key')!r} "
        f"bot_sid={bot_sid!r} display={bot_record.get('displayName')!r} "
        f"transcript={len(bot_transcript)}"
    )
    logger.info(
        "Dispatching bot turn: game=%s slot=%s bot_sid=%s display=%s transcript=%s roster_names=%s",
        game_key,
        bot_record.get("key"),
        bot_sid,
        bot_record.get("displayName"),
        len(bot_transcript),
        roster_names,
    )
    try:
        result = await bot_turn(
            bot_sid=bot_sid,
            transcript=bot_transcript,
            roster_names=roster_names,
            tools=get_bot_tools(game_key, bot_sid),
            tool_argument_overrides=get_bot_tool_argument_overrides(game_key, bot_sid),
        )
    except Exception as e:
        logger.exception("respond_as_bot failed")
        await atlantis.client_log(f"respond_as_bot failed: {type(e).__name__}: {e}")
        raise

    await atlantis.client_log(
        f"respond_as_bot done: bot_sid={bot_sid!r} chars={len(result or '')}"
    )
    return result
