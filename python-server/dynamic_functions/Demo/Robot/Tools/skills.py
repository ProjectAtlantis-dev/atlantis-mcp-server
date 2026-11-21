import logging
from pathlib import Path

logger = logging.getLogger("dynamic_function")

# One Markdown file per skill; the file name (without .md) is the skill's name
# and its first non-blank line is its description. Enumerated, never listed here.
SKILLS_PATH = Path(__file__).absolute().parent / "skills"


def _skill_files() -> dict[str, Path]:
    if not SKILLS_PATH.is_dir():
        raise FileNotFoundError(f"No skills folder at {SKILLS_PATH}")
    return {path.stem: path for path in sorted(SKILLS_PATH.glob("*.md"))}


def _skill_description(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return line.strip().lstrip("#").strip()
    raise ValueError(f"Skill file {path} is empty")


@visible
async def skill_list() -> list[dict]:
    """List the skills you can view, by name, with what each one is for."""
    logger.info("Executing skill_list...")
    return [{"name": name, "description": _skill_description(path)}
            for name, path in _skill_files().items()]


@visible
async def skill_view(name: str) -> str:
    """Show the full instructions of one skill, by a name from skill_list."""
    logger.info(f"Executing skill_view({name!r})...")
    skills = _skill_files()
    if name not in skills:
        raise ValueError(f"No skill named {name!r}; skills are: {', '.join(skills) or 'none'}")
    return skills[name].read_text(encoding="utf-8")
