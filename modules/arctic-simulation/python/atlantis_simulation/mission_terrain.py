"""Terrain supply worker; never integrates motion or owns a simulation tick.

The Node authority drives without an open browser. This worker supplies verified
nearby DEM/obstacles, and reports missing data explicitly instead of flat ground.
"""
import json
from .spatial_index import RectangleIndex
import logging
import math
import os
from pathlib import Path
import threading
import subprocess
import shutil
from urllib.parse import quote

from atlantis_economy.gateway import bank_request
from .terrain_adapter import configuration, elevation_grid, read_only, water_surface, flight_elevation_grid, catalog

log = logging.getLogger(__name__)


def make_surface(world, state, *, radius=None, step=None, center=None, allow_unknown=False, grid_anchor=None):
    from pyproj import Transformer
    config = configuration(world)
    frame = state["navigationFrame"]["origin"]
    airborne = state["authority"] in ("server-vtol-v1", "server-fixed-wing-v1")
    fixed = state["authority"] == "server-fixed-wing-v1"
    radius = radius if radius is not None else (768 if fixed else 64 if airborne else 256)
    step = step if step is not None else (8 if fixed else 4 if airborne else 2)
    center = center if center is not None else state["position"]
    grid = (water_surface(config, frame, state["position"]["z"], radius=radius, step=step, center=center, grid_anchor=grid_anchor)
            if state["authority"] == "server-boat-v1" else
            flight_elevation_grid(config, frame, water_level_m=state.get('mission', {}).get('waterLevelM'), radius=radius, step=step, center=center, allow_unknown=allow_unknown or fixed) if airborne else
            elevation_grid(config, frame, radius=radius, step=step, center=center, allow_unknown=allow_unknown or fixed))
    # Source building rings are EPSG:3413; conservatively pad bounding boxes by
    # 3m for the AMV footprint. Unknown source geometry is not ignored.
    to_map = Transformer.from_crs(4326, 3413, always_xy=True)
    to_geo = Transformer.from_crs(3413, 4326, always_xy=True)
    center_lon = frame["lon"] + center["x"]/6378137/math.cos(math.radians(frame["lat"]))*180/math.pi
    center_lat = frame["lat"] + center["y"]/6378137*180/math.pi
    cx, cy = to_map.transform(center_lon, center_lat)
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
    grid["roads"] = [] if state["authority"] == "server-boat-v1" else roads
    if state["authority"] == "server-boat-v1":
        grid["navigationDomain"] = "water"
        definition = catalog(config)["vehicle_definitions"][state["definitionId"]]
        length = definition["realLengthM"]
        profile = definition["boat"]
        radius = profile["maxSpeedMs"] * profile["yawDamping"] / profile["rudderTurnRadS2"]
        if not math.isfinite(length) or length <= 0 or not math.isfinite(radius) or radius <= 0:
            raise ValueError("Boat requires positive finite model length and turning radius")
        # Conservative circular hull envelope; preference adds room for a turn.
        grid["waterNavigation"] = {"hullRadiusM": length / 2,
                                   "preferredClearanceM": max(length, 2 * radius)}
    elif state["authority"] == "server-ground-v1":
        profiles = json.loads((Path(__file__).parent/"runtime/src/vehicle-performance.json").read_text())
        profile = profiles[state["definitionId"]]
        grades = profile["published"] if "published" in profile else profile["simulation"]
        grid["groundProfile"] = {"maxClimbingGrade": grades["climbingGradePercent"]/100,
                                 "maxSideGrade": grades["sideSlopePercent"]/100}
    if state["authority"] not in ("server-vtol-v1", "server-boat-v1"):
        grid["water"] = water_grid(config,grid)
    return grid


def destination_route(world, state):
    """Plan across the entire trip before releasing local vehicle controls.

    Unknown DEM/water cells are excluded, not filled with invented terrain.
    The bounded strategic lattice covers both endpoints plus a detour margin;
    local 2m grids still validate and control every driven segment.
    """
    start, target = state["position"], state["mission"]["target"]
    span = max(abs(target["x"]-start["x"]), abs(target["y"]-start["y"]))
    radius = span/2 + max(256, span/2)
    step = 2 * max(1, math.ceil(radius/1000))
    radius = math.ceil(radius/step)*step
    center = {axis:(start[axis]+target[axis])/2 for axis in ("x","y")}
    # Put the true departure on a strategic grid vertex: rounding to a nearby
    # coarse coastline cell must not relabel a verified-water start as land.
    surface = make_surface(world,state,radius=radius,step=step,center=center,allow_unknown=True,
                           grid_anchor=start if state['authority']=='server-boat-v1' else None)
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node is required for destination routing")
    result = subprocess.run([node,str(Path(__file__).parent/"runtime"/"src"/"plan-route.mjs")],
        input=json.dumps({"surface":surface,"start":start,"target":target}),
        text=True,capture_output=True,timeout=120)
    if result.returncode:
        raise RuntimeError("Destination routing failed: " + result.stderr[-1600:])
    return json.loads(result.stdout)


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
                        holding_fixed = state["authority"] == "server-fixed-wing-v1" and state.get("airborne") and mission and mission["status"] in ("completed", "awaiting_task")
                        if not mission or (mission["status"] not in ("queued", "running") and not holding_fixed):
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
                            refresh_distance = 8 if state["authority"] == "server-vtol-v1" else 32
                            if (mission["status"] != "queued" and state.get("missionTerrainReady") is True
                                    and last and math.hypot(position["x"]-last[0],position["y"]-last[1]) < refresh_distance):
                                continue
                            if state["authority"] in ("server-ground-v1", "server-boat-v1") and (not mission.get("destinationRoute") or mission.get("routeNeedsReplan")):
                                payload["destinationRoute"] = destination_route(world,state)
                            # Unknown cells stay impassable; one off-route hole must not discard the entire local patch.
                            payload["surface"] = make_surface(world,state,allow_unknown=True)
                        except subprocess.TimeoutExpired as error:
                            log.warning('Terrain mission %s route planning timed out: %s',mission['id'],error)
                            payload['error'] = 'route-planning-timeout: planner exceeded its time limit; no physical obstacle established'
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
            index=RectangleIndex(masks,lambda mask:mask[:4],(x0,y0,x1,y1))
            domain=[]
            for x,y in points:
                match=index.find(x,y)
                if match is None:domain.append(None);continue
                a,b,c,d,w,h,values=match
                ix=min(w-1,max(0,int((x-a)/(c-a)*(w-1))));iy=min(h-1,max(0,int((y-b)/(d-b)*(h-1))))
                domain.append(bool(values[iy*w+ix]))
            domains.append(domain)
    finally:
        connection.close()
    return [None if a is None or b is None else a or b for a,b in zip(*domains)]
