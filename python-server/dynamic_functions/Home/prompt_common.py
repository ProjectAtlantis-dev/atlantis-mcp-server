"""Shared system-prompt builder for game roles."""

import os
from datetime import datetime
from typing import List

from dynamic_functions.Home.common import home_path


def load_role_system_prompt(role: str) -> str:
    """Read the role's system_prompt.md and return its text."""
    path = os.path.join(home_path("Game", "Roles", role), "system_prompt.md")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_persona(bot_sid: str) -> str:
    """Read this bot's persona.md. Empty string if not provided."""
    path = os.path.join(home_path("Game", "Bots", bot_sid), "persona.md")
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
    persona: str = "",
    appearance: str = "",
    character_prompt: str = "",
    setting: str = "",
    caller: str = "",
    prior_interaction_count: int = 0,
    last_interaction_at: str = "",
    first_name: str = "",
) -> str:
    """Assemble the final system prompt: director's note + setting + persona + appearance + role base + character + time + interaction."""
    parts: List[str] = ["(director's note: we are striving for realistic dialog)"]
    if setting:
        parts.append(setting)
    if persona:
        parts.append(persona)
    if appearance:
        parts.append(f"Appearance: {appearance}")
    parts.append(base_prompt)
    if character_prompt:
        parts.append(character_prompt)
    parts.append(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    note = build_interaction_context(caller, prior_interaction_count, last_interaction_at, first_name)
    if note:
        parts.append(note)

    return "\n\n".join(parts)
