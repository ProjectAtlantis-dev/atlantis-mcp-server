import atlantis
import logging

logger = logging.getLogger("mcp_server")


@text("md")
@visible
async def README():
    """
    This contains general instructions on how to use this system
    """
    logger.info(f"Executing placeholder function: README...")

    await atlantis.client_log("README running")

    return """# Home

Welcome to your Home app on **Project Atlantis**!

## Getting Started

This is the default app that comes with your remote. You can:

- **Add functions** — drop `.py` files into `dynamic_functions/Home/` and they auto-load
- **Call tools** — use function names directly, or disambiguate with `**Home**function_name`
- **Install MCP servers** — add JSON configs to `dynamic_servers/`

## Example Function

```python
@visible
async def hello(name: str):
    \"\"\"Say hello to someone.\"\"\"
    return f"Hello, {name}!"
```

## Useful Tips

| Feature | Description |
|---------|-------------|
| `@visible` | Makes your function available as a tool |
| `@text("md")` | Returns output as rendered markdown |
| `@app(name="...")` | Legacy app assignment (just use folders instead) |

## Learn More

Visit [projectatlantis.ai](https://www.projectatlantis.ai) to manage your remotes and explore shared tools.
"""

