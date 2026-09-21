"""Reconcile saved town building bases against the finest verified stored DEM."""
import datetime
import json
import math
import numpy as np
from dynamic_functions.Terrain.dem_seams import read_continuous_dem


def plan_building_grounding(assets, terrain):
    tiles = terrain.execute("SELECT tile_id,x_min,y_min,x_max,y_max,depth FROM tiles "
                            "WHERE heightmap IS NOT NULL AND confidence_map IS NOT NULL "
                            "ORDER BY depth DESC,tile_id").fetchall()
    cache, decoded, updates, unresolved = {}, {}, [], []
    for asset_id, x, y, original in assets.execute(
            "SELECT id,cx,cy,properties FROM assets WHERE type='BYGNING' AND enabled=1"):
        props = json.loads(original)
        tile = next((t for t in tiles if t[1] <= x <= t[3] and t[2] <= y <= t[4]), None)
        if tile is None:
            unresolved.append({'id': asset_id, 'reason': 'no stored DEM coverage'})
            continue
        key,x0,y0,x1,y1,_ = tile
        if key not in decoded:
            decoded[key] = read_continuous_dem(terrain,key,cache)
        payload = decoded[key]
        if payload['vertical_datum'] != 'EGM2008':
            unresolved.append({'id':asset_id,'reason':'unverified vertical datum','tileId':key})
            continue
        values = payload['heightmap']; n = values.shape[0]
        gx,gy = (x-x0)/(x1-x0)*(n-1),(y-y0)/(y1-y0)*(n-1)
        ix,iy = min(n-2,int(gx)),min(n-2,int(gy)); u,v = gx-ix,gy-iy
        corners = values[iy:iy+2,ix:ix+2]
        if not np.isfinite(corners).all():
            unresolved.append({'id':asset_id,'reason':'unverified DEM sample','tileId':key})
            continue
        ground = float((corners[0,0]*(1-u)+corners[0,1]*u)*(1-v)+(corners[1,0]*(1-u)+corners[1,1]*u)*v)
        reference = {'source':'terrain-dem','tileId':key,'verticalDatum':'EGM2008',
                     'sampleSpacingM':(x1-x0)/(n-1),'updatedAt':payload['updated_at']}
        if props.get('groundReference') == reference and math.isclose(props['groundZ'],ground,abs_tol=1e-6):
            continue
        props.setdefault('originalGroundZ',props['groundZ'])
        props.update(groundZ=ground,groundSampled=True,groundReference=reference)
        updates.append({'id':asset_id,'original':original,'properties':json.dumps(props,separators=(',',':')),
                        'groundZ':ground,'previousGroundZ':json.loads(original)['groundZ'],'reference':reference})
    return updates, unresolved


def apply_building_grounding(assets, updates):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with assets:
        for item in updates:
            changed = assets.execute("UPDATE assets SET z=?,properties=?,updated_at=? WHERE id=? AND properties=?",
                (item['groundZ'],item['properties'],now,item['id'],item['original'])).rowcount
            if changed != 1:
                raise RuntimeError(f"Building changed during grounding: {item['id']}")
    return len(updates)
