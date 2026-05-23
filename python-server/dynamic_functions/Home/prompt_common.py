"""Shared system-prompt builder for game slots."""

import os
from datetime import datetime
from typing import List, Optional

from dynamic_functions.Home.common import home_path


def load_slot_system_prompt(slot: str) -> str:
    """Read the slot's system_prompt.md and return its text."""
    path = os.path.join(home_path("Game", "Slots", slot), "system_prompt.md")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()



def load_bot(bot_sid: str) -> str:
    """Read this bot's bot.md. Empty string if not provided."""
    path = os.path.join(home_path("Game", "Bots", bot_sid), "bot.md")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_appearance(bot_sid: str) -> str:
    """Read this bot's appearance.md. Empty string if not provided."""
    path = os.path.join(home_path("Game", "Bots", bot_sid), "appearance.md")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_interaction_context(
    caller: str,
    prior_interaction_count: int,
    last_interaction_at: str,
    first_name: str = "",
) -> str:
    """Generic note describing how well the bot knows this caller and when they last spoke."""
    if not caller:
        return ""

    display_name = first_name or caller

    hour = datetime.now().hour
    late_night = hour >= 22 or hour < 5

    if prior_interaction_count <= 0:
        if late_night:
            note = f"This is your first interaction with {display_name}. It's late — welcome them warmly."
        else:
            note = f"This is your first interaction with {display_name}. Introduce yourself and help them get oriented."
    elif prior_interaction_count == 1:
        note = f"You have interacted with {display_name} once before. Greet them like someone you remember, but don't overdo it."
    elif prior_interaction_count <= 5:
        note = f"You have interacted with {display_name} {prior_interaction_count} times before. They're still fairly new — keep context light."
    else:
        note = f"You have interacted with {display_name} {prior_interaction_count} times before. They're familiar — skip the intros and be casual."

    if last_interaction_at and prior_interaction_count > 0:
        try:
            elapsed = datetime.now() - datetime.fromisoformat(last_interaction_at)
            days = elapsed.days
            hours = elapsed.seconds // 3600
            if days > 30:
                note += f" It has been about {days // 30} month(s) since your last interaction — maybe acknowledge it's been a while."
            elif days > 0:
                note += f" It has been about {days} day(s) since your last interaction."
            elif hours > 0:
                note += f" Your last interaction was about {hours} hour(s) ago."
            else:
                note += " Your last interaction was moments ago."
        except (ValueError, TypeError):
            pass

    return note


def build_system_prompt(
    base_prompt: str,
    bot: str = "",
    appearance: str = "",
    character_prompt: str = "",
    setting: str = "",
    caller: str = "",
    prior_interaction_count: int = 0,
    last_interaction_at: str = "",
    first_name: str = "",
) -> str:
    """Assemble the final system prompt as markdown sections with titles."""
    parts: List[str] = ["## Director's Note\n\nWe are striving for realistic dialog."]
    if setting:
        parts.append(f"## Setting\n\n{setting}")
    if bot:
        parts.append(f"## Character\n\n{bot}")
    if appearance:
        parts.append(f"## Appearance\n\n{appearance}")
    parts.append(f"## Role\n\n{base_prompt}")
    if character_prompt:
        parts.append(f"## Casting Notes\n\n{character_prompt}")
    parts.append(f"## Current Time\n\n{datetime.now().strftime('%Y-%m-%d %H:%M')}")

    note = build_interaction_context(caller, prior_interaction_count, last_interaction_at, first_name)
    if note:
        parts.append(f"## Interaction History\n\n{note}")

    return "\n\n".join(parts)

@visible
def prompt_assemble(
    game_key: str,
    bot_sid: str,
    speaker_sid: str = "",
) -> str:
    """One-stop prompt assembly. Returns the full system-prompt string."""
    from dynamic_functions.Home.casting import (
        casting_for_occupant, load_casting_prompt, _slot_config, slot_for_occupant,
    )
    from dynamic_functions.Home.interactions import read_interaction
    from dynamic_functions.Home.location import compose_setting, position_get

    slot = slot_for_occupant(game_key, bot_sid)
    if not slot:
        raise ValueError(f"Bot {bot_sid} is not cast in any slot for game {game_key}")

    base_prompt = load_slot_system_prompt(slot)
    bot_md = load_bot(bot_sid)
    appearance_md = load_appearance(bot_sid)
    character_prompt = load_casting_prompt(slot, bot_sid)

    # Location / setting
    pos = position_get(game_key, bot_sid) or _slot_config(slot).get("defaultLocation", "")
    setting = compose_setting(pos) if pos else ""

    # Interaction history with speaker
    history = read_interaction(game_key, bot_sid, speaker_sid) if speaker_sid else {}

    # Resolve speaker's display name from casting records, fall back to history
    first_name = ""
    if speaker_sid:
        try:
            first_name = casting_for_occupant(game_key, speaker_sid).get("displayName", "") or ""
        except ValueError:
            first_name = ""
        first_name = first_name or history.get("first_name", "")

    return build_system_prompt(
        base_prompt=base_prompt,
        bot=bot_md,
        appearance=appearance_md,
        character_prompt=character_prompt,
        setting=setting,
        caller=speaker_sid,
        prior_interaction_count=int(history.get("count") or 0),
        last_interaction_at=history.get("last_interaction_at", ""),
        first_name=first_name,
    )
