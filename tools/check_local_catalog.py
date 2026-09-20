"""Verify the normal startup reader against the imported local catalog read-only."""
import ast
import json
from pathlib import Path
import sqlite3

root = Path(__file__).resolve().parents[1]
source = root / "python-server/dynamic_functions/Terrain/viewer_assets.py"
tree = ast.parse(source.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"_catalog_metadata", "_decoded_properties", "startup_assets"}]
path = root / "python-server/dynamic_functions/Terrain/Asset/assets.db"
class CatalogError(RuntimeError):
    pass
scope = {"Any": object, "sqlite3": sqlite3, "json": json,
         "AssetCatalogUnavailable": CatalogError,
         "_required_assets_db_path": lambda: path,
         "_connect_read_only": lambda p: sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
data = scope["startup_assets"]()
assert len(data["vehicle_instances"]) == 7
assert all(v["definitionId"] in data["vehicle_definitions"] for v in data["vehicle_instances"])
with sqlite3.connect(":memory:") as invalid:
    invalid.execute("CREATE TABLE asset_metadata(key TEXT, value TEXT)")
    for key, value in (("schema_version", 1), ("vehicle_asset_type", "vehicle"),
                       ("vehicle_definition", data["vehicle_definition"])):
        invalid.execute("INSERT INTO asset_metadata VALUES (?,?)", (key, json.dumps(value)))
    try:
        scope["_catalog_metadata"](invalid)
    except CatalogError:
        pass
    else:
        raise AssertionError("Incomplete non-multi catalog was silently accepted")
print("PASS: seven model bindings preserved; incomplete catalog rejected; original DB unchanged")
