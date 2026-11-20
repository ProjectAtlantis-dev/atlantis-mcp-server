import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@visible
async def list_demo_tools() -> str:
    """List every example tool in the Demo app, with what each one does."""
    # The path goes in the data section, so the shell never parses it
    rows = await atlantis.client_command("/ls", {"searchTerm": "$Demo/*"})
    if not isinstance(rows, list):
        raise TypeError(f"/ls returned {type(rows).__name__}, expected a list")

    # Example tools only: not subfolders (like Robot), the app's own main.py
    # (index, the walkthrough) or this listing
    lines = [
        f"- {row['name']}: {row['description']}"
        for row in rows
        if row.get("name")
        and row.get("filename") not in ("main.py", "list_demo_tools.py")
    ]
    if not lines:
        raise RuntimeError("/ls $Demo/* found no Demo tools")
    return "The Demo app's tools:\n" + "\n".join(lines)
