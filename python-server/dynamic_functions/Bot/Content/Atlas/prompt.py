"""Atlas's system prompt builder.

Assembles the final system prompt from the base prompt template
plus visitor-specific context.
"""

from datetime import datetime
from typing import List


def build_visitor_context(caller: str, visit_count: int, last_visit: str, first_name: str = "") -> str:
    """Build a visitor context note. Returns empty string if no context applies."""
    if not caller or visit_count <= 0:
        return ""

    display_name = first_name or caller

    hour = datetime.now().hour
    late_night = hour >= 22 or hour < 5

    if visit_count == 1:
        if late_night:
            visitor_note = f"{display_name} just arrived. It's late — welcome them warmly."
        else:
            visitor_note = f"{display_name} just arrived. They're brand new — introduce yourself, welcome them, and help them get oriented."
    elif visit_count <= 5:
        visitor_note = f"{display_name} has visited {visit_count} times. They're still fairly new — be friendly and remember they might still be figuring things out."
    else:
        visitor_note = f"{display_name} has visited {visit_count} times. They're a regular — skip the intros, be casual, and treat them like a friend."

    if last_visit and visit_count > 1:
        try:
            elapsed = datetime.now() - datetime.fromisoformat(last_visit)
            days = elapsed.days
            hours = elapsed.seconds // 3600
            if days > 30:
                visitor_note += f" It's been about {days // 30} month(s) since their last visit — maybe acknowledge it's been a while."
            elif days > 0:
                visitor_note += f" It's been about {days} day(s) since their last visit."
            elif hours > 0:
                visitor_note += f" They were here about {hours} hour(s) ago."
            else:
                visitor_note += " They were just here moments ago."
        except (ValueError, TypeError):
            pass

    return visitor_note


def build_system_prompt(
    base_prompt: str,
    caller: str = "",
    visit_count: int = 0,
    last_visit: str = "",
    first_name: str = ""
) -> str:
    parts: List[str] = [base_prompt]
    parts.append(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    visitor_note = build_visitor_context(caller, visit_count, last_visit, first_name)
    if visitor_note:
        parts.append(visitor_note)

    return "\n\n".join(parts)
