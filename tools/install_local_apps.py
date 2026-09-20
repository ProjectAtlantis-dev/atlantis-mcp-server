"""Replace known external app links with real checkout-local copies.

Run with the host stopped. Refuse actual directories (never overwrite user code).
The source code remains untouched. This is a one-time migration, not hot sync.
"""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
apps = root / "python-server/dynamic_functions"
sources = {"ArcticSimulation": root / "modules/arctic-simulation/dynamic_functions/ArcticSimulation"}
for name in ("GreenlandBank", "GreenlandGame", "GreenlandWarehouse"):
    sources[name] = root / "modules/bank-world/integrations/atlantis-mcp/dynamic_functions" / name

for name, source in sources.items():
    destination = apps / name
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not destination.is_symlink():
        raise RuntimeError(f"Expected original external app symlink: {destination}")

for name, source in sources.items():
    destination = apps / name
    staged = apps / (name + ".local-copy")
    shutil.copytree(source, staged, ignore=shutil.ignore_patterns("__pycache__"))
    destination.unlink()  # Remove only the verified link, never its source.
    staged.rename(destination)
    print(f"Installed real local directory: {name}")

venv_link = root / "python-server/venv"
if venv_link.is_symlink():
    venv_link.unlink()
    venv_link.symlink_to("../.venv", target_is_directory=True)
    print("venv now resolves within this checkout")
