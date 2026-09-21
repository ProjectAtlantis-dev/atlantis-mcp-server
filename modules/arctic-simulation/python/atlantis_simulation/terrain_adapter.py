"""Read-only Terrain catalog/DEM adapter; simulation and bank remain state owners.

No dependency on MCP host internals: a smaller host can use the same contract.
"""
from .controller_models import CONTROLLER_MODELS
from .spatial_index import RectangleIndex
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
from urllib.parse import quote
import zlib


def configuration(world):
    path = os.environ.get('ATLANTIS_TERRAIN_BINDINGS')
    if not path:
        raise RuntimeError('ATLANTIS_TERRAIN_BINDINGS must select existing catalog, terrain and UUID mappings')
    config = json.loads(Path(path).read_text())
    if config.get('version') != 1:
        raise ValueError('Unsupported Terrain binding version')
    return config['worlds'][world]


def read_only(path):
    resolved = Path(path).expanduser().resolve(strict=True)
    return sqlite3.connect(resolved.as_uri()+'?mode=ro', uri=True, timeout=5)


def guard_asset_write(asset_id):
    path=os.environ.get('ATLANTIS_TERRAIN_BINDINGS')
    if not path:
        return
    config=json.loads(Path(path).read_text())
    if config.get('version')!=1:
        raise ValueError('Invalid terrain authority configuration')
    for world in config['worlds'].values():
        controlled = {key: value for key, value in world['vehicles'].items() if value.get('controlEnabled', True)}
        if asset_id in controlled or any(v['terrainAssetId']==asset_id for v in controlled.values()):
            raise ValueError('Server-owned terrain vehicle: use authenticated vehicle controls, not catalog pose writes')


def catalog(config):
    connection = read_only(config['assetDatabase'])
    try:
        metadata = {key: json.loads(value) for key, value in connection.execute('SELECT key,value FROM asset_metadata')
                    if key in ('vehicle_definitions','vehicle_definition','vehicle_asset_type','schema_version')}
        definitions = metadata.get('vehicle_definitions')
        if not isinstance(definitions, dict) or not definitions:
            raise ValueError('Explicit multi-vehicle definitions required; no AMV fallback for unknown models')
        instances = []
        for row in connection.execute('SELECT id,lat,lon,heading_deg,z,properties,saved_at FROM assets WHERE enabled=1 AND type=? ORDER BY id',
                                      (metadata['vehicle_asset_type'],)):
            props = json.loads(row[5]); definition_id = props.get('definitionId')
            if definition_id not in definitions:
                raise ValueError(f'Unknown vehicle definition for {row[0]}')
            instances.append(dict(props, id=row[0], lat=row[1], lon=row[2], headingDeg=row[3], z=row[4], savedAt=row[6]))
        return {'source':'terrain_asset_catalog','schemaVersion':metadata['schema_version'],
                'vehicle_definition':metadata['vehicle_definition'],'vehicle_definitions':definitions,'vehicle_instances':instances}
    finally:
        connection.close()


def elevation_grid(config, instance, radius=20, step=2, center=None, allow_unknown=False):
    # Projection is specific to Terrain's published EPSG:3413 database contract.
    from pyproj import Transformer
    transform = Transformer.from_crs(4326, 3413, always_xy=True)
    connection = read_only(config['terrainDatabase'])
    heights, sources, cache = [], {}, {}
    dem_cache = {}
    count = round(2*radius/step)+1
    lat, lon = instance['lat'], instance['lon']
    offset_x, offset_y = (center['x'], center['y']) if center is not None else (0, 0)
    # Keep overlapping patches on one lattice: refreshing must not change slopes.
    offset_x, offset_y = round(offset_x/step)*step, round(offset_y/step)*step
    try:
        projected=[]
        for east in (offset_x-radius,offset_x,offset_x+radius):
            for north in (offset_y-radius,offset_y,offset_y+radius):
                projected.append(transform.transform(lon+east/(6378137*math.cos(math.radians(lat)))*180/math.pi,lat+north/6378137*180/math.pi))
        x0=min(p[0] for p in projected)-1; x1=max(p[0] for p in projected)+1
        y0=min(p[1] for p in projected)-1; y1=max(p[1] for p in projected)+1
        tiles=connection.execute("SELECT tile_id,x_min,y_min,x_max,y_max,source,updated_at FROM tiles "
            "WHERE x_min<=? AND x_max>=? AND y_min<=? AND y_max>=? "
            "AND vertical_datum='EGM2008' AND heightmap IS NOT NULL AND confidence_map IS NOT NULL AND depth>=12 "
            "ORDER BY depth DESC",(x1,x0,y1,y0)).fetchall()
        tile_index=RectangleIndex(tiles,lambda tile:tile[1:5],(x0,y0,x1,y1))
        for row in range(count):
            for col in range(count):
                east, north = offset_x-radius+col*step, offset_y-radius+row*step
                sample_lat = lat+north/6378137*180/math.pi
                sample_lon = lon+east/(6378137*math.cos(math.radians(lat)))*180/math.pi
                x, y = transform.transform(sample_lon, sample_lat)
                tile = tile_index.find(x,y)
                if tile is None:
                    if allow_unknown:
                        heights.append(None)
                        continue
                    raise ValueError(f'No verified EGM2008 DEM at {sample_lat},{sample_lon}; acquire terrain before attaching')
                key,x0,y0,x1,y1,source,updated = tile
                if key not in cache:
                    from dynamic_functions.Terrain.dem_seams import read_continuous_dem
                    payload = read_continuous_dem(connection, key, dem_cache)
                    if payload is None:
                        raise ValueError(f'DEM disappeared while sampling {key}')
                    values = payload['heightmap']
                    n = values.shape[0]
                    cache[key] = (n, values.ravel().tolist(), payload['confidence_map'].ravel())
                n,values,confidence = cache[key]
                gx,gy = (x-x0)/(x1-x0)*(n-1),(y-y0)/(y1-y0)*(n-1)
                ix,iy = min(n-2,int(gx)),min(n-2,int(gy)); tx,ty=gx-ix,gy-iy
                indices = [iy*n+ix,iy*n+ix+1,(iy+1)*n+ix,(iy+1)*n+ix+1]
                if any(not confidence[i] or not math.isfinite(values[i]) for i in indices):
                    if allow_unknown:
                        heights.append(None)
                        continue
                    raise ValueError(f'Missing/confidence-zero ground sample in {key}')
                a,b,c,d = [values[i] for i in indices]
                heights.append((a*(1-tx)+b*tx)*(1-ty)+(c*(1-tx)+d*tx)*ty)
                sources[key] = {'source':source,'updatedAt':updated}
    finally:
        connection.close()
    digest = hashlib.sha256(json.dumps({'heights':heights,'sources':sources},sort_keys=True).encode()).hexdigest()
    return {'id':'terrain-dem:'+digest,'origin':{'lat':lat,'lon':lon},'minX':offset_x-radius,'minY':offset_y-radius,
            'stepM':step,'rows':count,'cols':count,'heights':heights,'verticalDatum':'EGM2008','sources':sources}


def water_surface(config, origin, water_level_m, radius=256, step=2, center=None, grid_anchor=None):
    """Water navigation samples coastline masks, not a fabricated seabed DEM."""
    from .mission_terrain import water_grid
    if not math.isfinite(water_level_m):
        raise ValueError('Explicit water level required')
    center = center or {"x": 0, "y": 0}
    anchor = grid_anchor if grid_anchor is not None else {"x": 0, "y": 0}
    if any(not math.isfinite(anchor[axis]) for axis in ("x", "y")):
        raise ValueError('Finite water grid anchor required')
    count = round(2*radius/step)+1
    grid = {"origin": {"lat": origin["lat"], "lon": origin["lon"]},
            "minX": anchor["x"] + round((center["x"]-anchor["x"])/step)*step-radius,
            "minY": anchor["y"] + round((center["y"]-anchor["y"])/step)*step-radius,
            "stepM": step, "rows": count, "cols": count,
            "heights": [water_level_m]*(count*count),
            "navigationDomain": "water", "verticalDatum": "configured-water-level"}
    grid["water"] = water_grid(config, grid)
    grid["id"] = "terrain-water:" + hashlib.sha256(json.dumps(grid, sort_keys=True).encode()).hexdigest()
    return grid


def attachment(world, asset_id):
    config = configuration(world)
    mapping = config['vehicles'][asset_id]
    aliases = [v['terrainAssetId'] for v in config['vehicles'].values()]
    if len(aliases)!=len(set(aliases)):
        raise ValueError('A terrain vehicle cannot map to two canonical UUIDs')
    assets = catalog(config)
    instance = next((v for v in assets['vehicle_instances'] if v['id']==mapping['terrainAssetId']),None)
    if instance is None:
        raise ValueError('Mapped existing terrain vehicle not found')
    if instance['definitionId'] not in CONTROLLER_MODELS:
        raise ValueError('No authoritative controller for this catalog model')
    surface = water_surface(config, instance, instance["z"], radius=20) if CONTROLLER_MODELS[instance["definitionId"]] == "boat" else elevation_grid(config,instance)
    return {'terrainAssetId':instance['id'],'definitionId':instance['definitionId'],
            'definition':assets['vehicle_definitions'][instance['definitionId']],
            'pose':{'x':0,'y':0,'headingRad':math.radians(instance['headingDeg'])},'surface':surface,
            'sourcePose':{'lat':instance['lat'],'lon':instance['lon'],'z':instance['z'],'headingDeg':instance['headingDeg']}}


def startup(world, snapshot, owned_assets=()):
    config=configuration(world);payload=catalog(config)
    controlled={v['terrainAssetId']:v for v in snapshot.get('controlledVehicles',[])}
    ownership = {}
    for asset in owned_assets:
        metadata = asset.get('metadata', {})
        if asset.get('kind') != 'vehicle' or metadata.get('world') != world:
            continue
        source_id = metadata.get('terrainAssetId')
        if not source_id:
            continue
        if source_id in ownership:
            raise ValueError('Multiple bank UUIDs identify one Terrain vehicle')
        ownership[source_id] = asset
    for item in payload['vehicle_instances']:
        source_id = item['id']
        owner = ownership.get(source_id)
        if owner is not None:
            if owner['assetType'] != item['definitionId']:
                raise ValueError('Bank/Terrain vehicle model conflict')
            item.update(id=owner['id'], bankAssetId=owner['id'], terrainAssetId=source_id,
                        ownerAccountId=owner['ownerAccountId'], ownerUsername=owner['ownerName'])
        state=controlled.get(source_id)
        if state is None:
            continue
        binding=config['vehicles'].get(state['id'])
        if not binding or binding['terrainAssetId']!=source_id or state['definitionId']!=item['definitionId']:
            raise ValueError('Terrain/simulation identity or model conflict')
        if owner is not None and (owner['id'] != state['id'] or owner['ownerAccountId'] != state['ownerAccountId']):
            raise ValueError('Bank/simulation vehicle ownership conflict')
        legacy=source_id
        item.update(id=state['id'],terrainAssetId=legacy,lat=state['lat'],lon=state['lon'],z=state['position']['z'],
                    headingDeg=math.degrees(state['headingRad']),authority=state['authority'])
    payload['defense_definition']={'authority':{'mode':'remote','gameId':world,'snapshotUrl':f'/api/simulation/{quote(world,safe="")}/snapshot'}}
    payload['owned_infrastructure'] = [
        {'id': asset['id'], 'modelId': asset['assetType'], 'ownerUsername': asset['ownerName']}
        for asset in owned_assets
        if asset.get('kind') == 'structure' and asset.get('status') == 'active'
        and asset.get('metadata', {}).get('world') == world
        and asset.get('metadata', {}).get('stateAuthority') == 'arctic-simulation'
    ]
    payload['simulation_snapshot']=copy.deepcopy(snapshot)
    return payload


def flight_elevation_grid(config, origin, water_level_m=None, **kwargs):
    if water_level_m is None:
        return elevation_grid(config, origin, **kwargs)
    if type(water_level_m) not in (int, float) or not math.isfinite(water_level_m):
        raise ValueError('Flight water level must be finite')
    from .mission_terrain import water_grid
    kwargs['allow_unknown'] = True
    grid = elevation_grid(config, origin, **kwargs)
    water = water_grid(config, grid)
    grid['heights'] = [float(water_level_m) if wet is True else height
                       for wet, height in zip(water, grid['heights'])]
    grid['water'] = water
    grid['waterLevelM'] = water_level_m
    grid['id'] = 'flight-surface:' + hashlib.sha256(json.dumps(grid, sort_keys=True).encode()).hexdigest()
    return grid
