"""Terrain supply worker; never integrates motion or owns a simulation tick.

The Node authority drives without an open browser. This worker supplies verified
nearby DEM/obstacles, and reports missing data explicitly instead of flat ground.
"""
import json
import logging
import math
import os
from pathlib import Path
import threading
from urllib.parse import quote

from atlantis_economy.gateway import bank_request
from .terrain_adapter import configuration, elevation_grid, read_only

log = logging.getLogger(__name__)


def make_surface(world, state):
    from pyproj import Transformer
    config = configuration(world)
    frame = state["navigationFrame"]["origin"]
    airborne = state["authority"] == "server-vtol-v1"
    radius = 64 if airborne else 128
    grid = elevation_grid(config, frame, radius=radius, step=4 if airborne else 2, center=state["position"])
    # Source building rings are EPSG:3413; conservatively pad bounding boxes by
    # 3m for the AMV footprint. Unknown source geometry is not ignored.
    to_map = Transformer.from_crs(4326, 3413, always_xy=True)
    to_geo = Transformer.from_crs(3413, 4326, always_xy=True)
    cx, cy = to_map.transform(state["lon"], state["lat"])
    query_radius = radius * 1.5 + 20
    obstacles = []
    roads = []
    connection = read_only(config["assetDatabase"])
    try:
        rows = connection.execute("SELECT id,properties FROM assets WHERE enabled=1 AND type='BYGNING' "
            "AND min_x<=? AND max_x>=? AND min_y<=? AND max_y>=?", (cx+query_radius,cx-query_radius,cy+query_radius,cy-query_radius))
        for asset_id, properties in rows:
            ring = json.loads(properties).get("ring")
            if not ring:
                raise ValueError(f"Building {asset_id} has no collision footprint")
            points = [to_geo.transform(p[0], p[1]) for p in ring]
            xs = [(lon-frame["lon"])*math.pi/180*6378137*math.cos(math.radians(frame["lat"])) for lon, lat in points]
            ys = [(lat-frame["lat"])*math.pi/180*6378137 for lon, lat in points]
            obstacles.append({"id":asset_id,"minX":min(xs)-3,"maxX":max(xs)+3,"minY":min(ys)-3,"maxY":max(ys)+3,"maxZ":max(p[2] for p in ring)})
        rows = connection.execute("SELECT id,properties FROM assets WHERE enabled=1 AND type='VEJMIDTE' "
            "AND min_x<=? AND max_x>=? AND min_y<=? AND max_y>=?", (cx+query_radius,cx-query_radius,cy+query_radius,cy-query_radius))
        for asset_id, properties in rows:
            path = json.loads(properties).get("path")
            if not path or len(path) < 2:
                raise ValueError(f"Road {asset_id} has no centerline geometry")
            points = [to_geo.transform(p[0], p[1]) for p in path]
            roads.append({"id":asset_id,"path":[{
                "x":(lon-frame["lon"])*math.pi/180*6378137*math.cos(math.radians(frame["lat"])),
                "y":(lat-frame["lat"])*math.pi/180*6378137} for lon,lat in points]})
    finally:
        connection.close()
    grid["obstacles"] = obstacles
    grid["roads"] = roads
    if state["authority"] != "server-vtol-v1":
        grid["water"] = water_grid(config,grid)
    return grid


class MissionTerrainWorker:
    def __init__(self, host):
        self.host = host
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="mission-terrain-supply")
        self.last = {}

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()

    def run(self):
        while not self.stopped.wait(.5):
            try:
                policy_path = os.environ.get("ATLANTIS_TERRAIN_BINDINGS")
                if not policy_path:
                    continue
                worlds = json.loads(Path(policy_path).read_text())["worlds"]
                for world in worlds:
                    if self.stopped.is_set():
                        return
                    path = f'/games/{quote(world, safe="")}'
                    snapshot = self.host.command("GET", path+"/snapshot")
                    for state in snapshot.get("controlledVehicles", []):
                        mission = state.get("mission")
                        if not mission or mission["status"] not in ("queued", "running"):
                            continue
                        key = (world,state["id"],mission["id"])
                        position = state["position"]
                        last = self.last.get(key)
                        # Revalidate bank ownership even when no new DEM is needed.
                        payload = {"operation":"mission_surface","id":state["id"],"missionId":mission["id"]}
                        try:
                            verified = bank_request("GET", f'/assets/{state["id"]}/verify')
                            asset = verified.get("asset") or {}
                            if (not verified.get("authentic") or not verified.get("spendable")
                                    or asset.get("ownerAccountId") != state["ownerAccountId"]
                                    or asset.get("metadata", {}).get("world") != world
                                    or asset.get("assetType") != state["definitionId"]):
                                raise PermissionError("bank ownership/authority no longer valid")
                            if mission["status"] != "queued" and last and math.hypot(position["x"]-last[0],position["y"]-last[1]) < 8:
                                continue
                            payload["surface"] = make_surface(world,state)
                        except Exception as error:
                            log.warning("Terrain mission %s blocked: %s",mission["id"],error)
                            payload["error"] = "terrain-or-ownership-unavailable: " + str(error)[:200]
                        if self.stopped.is_set():
                            return
                        result = self.host.command("POST", path+"/vehicle-control", payload)
                        if result.get("error"):
                            raise RuntimeError(result.get("message", result["error"]))
                        self.last[key] = (position["x"],position["y"])
            except Exception:
                log.exception("Mission terrain supply failed; vehicles remain bounded by verified coverage")


def water_grid(config, grid):
    """Read south-first coastal and inland masks; unknown coverage stays blocked."""
    import zlib
    from pyproj import Transformer
    transform = Transformer.from_crs(4326,3413,always_xy=True)
    origin = grid['origin']
    points=[]
    for row in range(grid['rows']):
        for col in range(grid['cols']):
            lon=origin['lon']+(grid['minX']+col*grid['stepM'])/6378137/math.cos(math.radians(origin['lat']))*180/math.pi
            lat=origin['lat']+(grid['minY']+row*grid['stepM'])/6378137*180/math.pi
            points.append(transform.transform(lon,lat))
    x0=min(p[0] for p in points);x1=max(p[0] for p in points);y0=min(p[1] for p in points);y1=max(p[1] for p in points)
    connection=read_only(config['terrainDatabase'])
    domains=[]
    try:
        for table in ('coastline_masks','hydrography_masks'):
            masks=[]
            for row in connection.execute(f'SELECT t.x_min,t.y_min,t.x_max,t.y_max,m.width,m.height,m.mask FROM {table} m JOIN tiles t ON t.tile_id=m.tile_id WHERE t.x_min<=? AND t.x_max>=? AND t.y_min<=? AND t.y_max>=? ORDER BY t.depth DESC',(x1,x0,y1,y0)):
                *bounds,width,height,blob=row
                values=zlib.decompress(blob)
                if len(values)!=width*height:raise ValueError('Invalid stored water mask')
                masks.append((*bounds,width,height,values))
            domain=[]
            for x,y in points:
                match=next((m for m in masks if m[0]<=x<=m[2] and m[1]<=y<=m[3]),None)
                if match is None:domain.append(None);continue
                a,b,c,d,w,h,values=match
                ix=min(w-1,max(0,int((x-a)/(c-a)*(w-1))));iy=min(h-1,max(0,int((y-b)/(d-b)*(h-1))))
                domain.append(bool(values[iy*w+ix]))
            domains.append(domain)
    finally:
        connection.close()
    return [None if a is None or b is None else a or b for a,b in zip(*domains)]
