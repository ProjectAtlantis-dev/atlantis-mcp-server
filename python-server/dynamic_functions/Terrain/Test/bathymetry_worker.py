"""Isolated regression gates for the bundled bathymetry acquisition runtime."""

from contextlib import closing
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import zlib

import numpy as np

from dynamic_functions.Terrain.Bathymetry import worker
from dynamic_functions.Terrain import bathymetry_demand
from dynamic_functions.Terrain.Bathymetry.inputs import job_regions, missing_inputs
from dynamic_functions.Terrain.Database import schema
from dynamic_functions.Terrain.Database.tiles import ensure_tile_row
from dynamic_functions.Terrain.demand import DemandCoordinator, DemandLane
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import tile_bounds

JOB = "8-100-100"


def _source_digest(connection):
    digest = hashlib.sha256()
    for table in ("tiles", "coastline_masks"):
        digest.update(repr(connection.execute(f"SELECT * FROM {table} ORDER BY tile_id").fetchall()).encode())
    return digest.hexdigest()


class BathymetryWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="terrain-bathymetry-test-")
        cls.path = Path(cls.directory.name) / "terrain.db"
        bounds = tile_bounds(JOB, GREENLAND_BBOX)
        centre_x = (bounds[0] + bounds[2]) / 2
        with closing(sqlite3.connect(cls.path)) as connection:
            schema.create(connection)
            for depth, rect in job_regions(JOB).items():
                for col in range(rect[0], rect[1] + 1):
                    for row in range(rect[2], rect[3] + 1):
                        tid = f"{depth}-{col}-{row}"
                        x0, y0, x1, y1 = tile_bounds(tid, GREENLAND_BBOX)
                        xs = np.linspace(x0, x1, 65)[None, :]
                        ys = np.linspace(y0, y1, 65)[:, None]
                        distance = np.abs(xs - centre_x - 200 * np.sin(ys / 3000))
                        wet = distance < 1100
                        dem = np.where(wet, 0, (distance - 1100) * 0.3 + 2).astype(np.float32)
                        ensure_tile_row(connection, tid)
                        connection.execute("UPDATE tiles SET heightmap=?,source='fixture',vertical_datum='EGM2008' WHERE tile_id=?",
                                           (zlib.compress(dem.tobytes()), tid))
                        connection.execute("INSERT INTO coastline_masks (tile_id,width,height,mask,source,version,updated_at) VALUES (?,65,65,?,'fixture',2,'fixture')",
                                           (tid, zlib.compress(wet.astype(np.uint8).tobytes())))
            connection.commit()
            cls.before = _source_digest(connection)
        cls.result = worker.run(cls.path, JOB)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_full_job_preserves_sources_and_shoreline(self):
        self.assertTrue(self.result["written"])
        self.assertGreater(self.result["rows"], 1)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(_source_digest(connection), self.before)
            rows = connection.execute("SELECT b.tile_id,b.heightmap,t.heightmap,c.mask FROM bathymetry b JOIN tiles t USING(tile_id) JOIN coastline_masks c USING(tile_id)").fetchall()
            self.assertIn(JOB, {row[0] for row in rows})
            for tid, encoded, original, mask in rows:
                bed = np.frombuffer(zlib.decompress(encoded), dtype=np.float32).reshape(65, 65)
                dem = np.frombuffer(zlib.decompress(original), dtype=np.float32).reshape(65, 65)
                wet = np.frombuffer(zlib.decompress(mask), dtype=np.uint8).reshape(65, 65).astype(bool)
                np.testing.assert_array_equal(bed[~wet], dem[~wet], err_msg=tid)
                shore = worker.ndimage.binary_dilation(~wet, structure=np.ones((3, 3))) & wet
                np.testing.assert_array_equal(bed[shore & np.isfinite(bed)], 0, err_msg=tid)
            root = next(row for row in rows if row[0] == JOB)
            self.assertLess(np.nanmin(np.frombuffer(zlib.decompress(root[1]), dtype=np.float32)), -10)

    def test_cli_is_self_contained_and_idempotent(self):
        server_dir = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            [sys.executable, "-m", "dynamic_functions.Terrain.Bathymetry.worker", "--db", str(self.path), "--tile", JOB],
            cwd=self.directory.name, env={**os.environ, "PYTHONPATH": str(server_dir)},
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(result.stdout), {"tileId": JOB, "written": False, "rows": 0})

    def test_input_coverage_includes_halo_and_all_water_can_omit_dem(self):
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertFalse(any(missing_inputs(connection, JOB).values()))
            tid = "12-1599-1599"  # southwest apron, outside the core region
            connection.execute("DELETE FROM coastline_masks WHERE tile_id=?", (tid,))
            self.assertIn(tid, missing_inputs(connection, JOB)["coastline"])
            connection.rollback()
            tid = connection.execute("SELECT tile_id FROM tiles LIMIT 1").fetchone()[0]
            connection.execute("UPDATE tiles SET heightmap=NULL WHERE tile_id=?", (tid,))
            connection.execute("UPDATE coastline_masks SET mask=? WHERE tile_id=?",
                               (zlib.compress(np.ones((65, 65), dtype=np.uint8).tobytes()), tid))
            self.assertNotIn(tid, missing_inputs(connection, JOB)["dem"])

    def test_publication_failure_rolls_back_every_row(self):
        path = Path(self.directory.name) / "failure.db"
        with closing(sqlite3.connect(path)) as connection:
            schema.create(connection)
            connection.execute(f"CREATE TRIGGER reject_job BEFORE INSERT ON bathymetry WHEN NEW.tile_id='{JOB}' BEGIN SELECT RAISE(ABORT, 'fixture publication failed'); END")
        rows = [("9-200-200", np.full((65, 65), -10, dtype=np.float32), 4225, worker.SOURCE),
                (JOB, np.full((65, 65), -10, dtype=np.float32), 4225, worker.LOD_SOURCE)]
        with patch.object(worker, "build", return_value=rows):
            with self.assertRaisesRegex(sqlite3.IntegrityError, "fixture publication failed"):
                worker.run(path, JOB)
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM bathymetry").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM tiles").fetchone()[0], 0)

    def test_camera_polls_retain_only_current_job_prerequisites(self):
        lanes = {name: Mock() for name in ("dem", "coastline", "bathymetry")}
        coordinator = DemandCoordinator(lanes)
        coordinator.refresh({"dem": ["camera"], "bathymetry": [JOB]})
        coordinator.submit_bathymetry_inputs(JOB, {"dem": ["halo"], "coastline": ["mask"]})
        coordinator.refresh({"dem": ["camera"], "bathymetry": [JOB]})
        lanes["dem"].replace_pending.assert_called_with(["camera"], auxiliary_ids=["halo"])
        lanes["coastline"].replace_pending.assert_called_with([], auxiliary_ids=["mask"])
        coordinator.refresh({"dem": ["new-camera"], "bathymetry": []})
        lanes["dem"].replace_pending.assert_called_with(["new-camera"])
        lanes["coastline"].replace_pending.assert_called_with([])
        lanes["dem"].submit.reset_mock()
        coordinator.submit_bathymetry_inputs(JOB, {"dem": ["obsolete"]})
        lanes["dem"].submit.assert_not_called()

    def test_missing_inputs_converge_through_real_lanes_and_local_process(self):
        path = Path(self.directory.name) / "demand.db"
        with closing(sqlite3.connect(self.path)) as source, closing(sqlite3.connect(path)) as connection:
            source.backup(connection)
            connection.execute("DELETE FROM bathymetry")
            dem_id, mask_id = "12-1599-1599", "11-796-796"
            dem = connection.execute("SELECT heightmap FROM tiles WHERE tile_id=?", (dem_id,)).fetchone()[0]
            coast = connection.execute("SELECT * FROM coastline_masks WHERE tile_id=?", (mask_id,)).fetchone()
            connection.execute("UPDATE tiles SET heightmap=NULL WHERE tile_id=?", (dem_id,))
            connection.execute("DELETE FROM coastline_masks WHERE tile_id=?", (mask_id,))
            connection.commit()
        released = threading.Event()
        now = [100.0]

        def acquire(tid):
            if not released.wait(timeout=5):
                raise TimeoutError("fixture provider was not released")
            with closing(sqlite3.connect(path)) as connection:
                with connection:
                    if tid == dem_id:
                        connection.execute("UPDATE tiles SET heightmap=? WHERE tile_id=?", (dem, tid))
                    elif tid == mask_id:
                        connection.execute("INSERT INTO coastline_masks VALUES (?,?,?,?,?,?,?,?)", coast)
                    else:
                        raise AssertionError(f"unexpected prerequisite: {tid}")
            return {"tileId": tid, "written": True}

        lanes = {
            "dem": DemandLane("fixture-bathy-dem", acquire, 1),
            "coastline": DemandLane("fixture-bathy-coast", acquire, 1),
            "bathymetry": DemandLane("fixture-bathy", bathymetry_demand.run_bathymetry_job,
                                      1, retry_delays=(1.0,), clock=lambda: now[0]),
        }
        coordinator = DemandCoordinator(lanes)
        with (
            closing(sqlite3.connect(path, check_same_thread=False)) as connection,
            patch.object(bathymetry_demand, "db", return_value=connection),
            patch.object(bathymetry_demand, "DATABASE_PATH", path),
            patch("dynamic_functions.Terrain.demand._coordinator", return_value=coordinator),
        ):
            try:
                coordinator.refresh({"bathymetry": [JOB]})
                self.assertTrue(lanes["bathymetry"].wait_for_idle(timeout=5))
                failure = lanes["bathymetry"].status()["failures"][JOB]
                self.assertTrue(failure["retryable"])
                self.assertIn("coverage incomplete", failure["error"])
                coordinator.refresh({"bathymetry": [JOB]})
                released.set()
                for name in ("dem", "coastline"):
                    self.assertTrue(lanes[name].wait_for_idle(timeout=5))
                now[0] = 102.0
                coordinator.refresh({"bathymetry": [JOB]})
                self.assertTrue(lanes["bathymetry"].wait_for_idle(timeout=45))
                self.assertFalse(lanes["bathymetry"].status()["failures"])
                for name in ("dem", "coastline"):
                    self.assertTrue(lanes[name].wait_for_idle(timeout=5))
                    self.assertFalse(lanes[name].status()["failures"])
                    self.assertEqual(lanes[name].status()["totals"]["started"], 1)
                self.assertIsNotNone(connection.execute("SELECT 1 FROM bathymetry WHERE tile_id=?", (JOB,)).fetchone())
                self.assertEqual(_source_digest(connection), self.before)
            finally:
                released.set()
                for lane in lanes.values():
                    lane.close()

    def test_invalid_job_addresses_are_rejected(self):
        for tid in ("9-100-100", "8-256-0", "8-0-256", "08-100-100"):
            with self.subTest(tile_id=tid), self.assertRaises(ValueError):
                job_regions(tid)


@visible
def bathymetry_worker_offline() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BathymetryWorkerTest)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(output.getvalue())
    return {"passed": True, "testsRun": result.testsRun}
