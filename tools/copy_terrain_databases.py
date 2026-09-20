"""One-time, non-overwriting SQLite import; includes committed WAL contents.

Run while the destination host is stopped. Source files are opened read-only.
Paths supplied here are import provenance, never runtime dependencies.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import tempfile


def copy_database(source, destination):
    source = source.resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sqlite-import-", dir=destination.parent) as temp:
        staged = Path(temp) / destination.name
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
            with sqlite3.connect(staged) as copied:
                original.backup(copied)
                check = copied.execute("PRAGMA integrity_check").fetchall()
                if check != [("ok",)]:
                    raise RuntimeError(f"Database integrity failed: {check}")
                tables = copied.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                counts = {}
                for (name,) in tables:
                    quoted = '"' + name.replace('"', '""') + '"'
                    counts[name] = copied.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
        # Hard link atomically refuses an existing target, unlike rename/replace.
        destination.hardlink_to(staged)
    return {"source": str(source), "destination": str(destination), "rows": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-terrain", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / "python-server/dynamic_functions/Terrain"
    records = []
    for relative in ("Database/terrain.db", "Asset/assets.db"):
        records.append(copy_database(args.source_terrain / relative, destination / relative))
    print(json.dumps(records, indent=2))
