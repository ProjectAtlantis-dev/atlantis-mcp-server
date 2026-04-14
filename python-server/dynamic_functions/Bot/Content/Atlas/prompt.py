"""Atlas's system prompt builder.

Assembles the final system prompt from the base prompt template
plus bot-specific interaction context.
"""

from datetime import datetime
from typing import List


def build_interaction_context(
    caller: str,
    prior_interaction_count: int,
    last_interaction_at: str,
    first_name: str = "",
) -> str:
    """Build a bot-specific interaction note."""
    if not caller:
        return ""

    display_name = first_name or caller

    hour = datetime.now().hour
    late_night = hour >= 22 or hour < 5

    if prior_interaction_count <= 0:
        if late_night:
            interaction_note = f"This is your first interaction with {display_name}. It's late; welcome them warmly and keep the tone calm."
        else:
            interaction_note = f"This is your first interaction with {display_name}. Be warm and welcoming; introduce yourself only if it fits naturally."
    elif prior_interaction_count == 1:
        interaction_note = f"You have interacted with {display_name} once before. Welcome them back without doing a full first-time intro."
    elif prior_interaction_count <= 5:
        interaction_note = f"You have interacted with {display_name} {prior_interaction_count} times before. They're still fairly new — be friendly and keep context light."
    else:
        interaction_note = f"You have interacted with {display_name} {prior_interaction_count} times before. They're familiar with you — skip the intros and be direct."

    if last_interaction_at and prior_interaction_count > 0:
        try:
            elapsed = datetime.now() - datetime.fromisoformat(last_interaction_at)
            days = elapsed.days
            hours = elapsed.seconds // 3600
            if days > 30:
                interaction_note += f" It has been about {days // 30} month(s) since your last interaction — maybe acknowledge it's been a while."
            elif days > 0:
                interaction_note += f" It has been about {days} day(s) since your last interaction."
            elif hours > 0:
                interaction_note += f" Your last interaction was about {hours} hour(s) ago."
            else:
                interaction_note += " Your last interaction was moments ago."
        except (ValueError, TypeError):
            pass

    return interaction_note


def build_system_prompt(
    base_prompt: str,
    caller: str = "",
    prior_interaction_count: int = 0,
    last_interaction_at: str = "",
    first_name: str = ""
) -> str:
    parts: List[str] = [base_prompt]
    parts.append(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    interaction_note = build_interaction_context(
        caller,
        prior_interaction_count,
        last_interaction_at,
        first_name,
    )
    if interaction_note:
        parts.append(interaction_note)

    return "\n\n".join(parts)
