"""Offline acquisition-date contract tests; only temporary databases are used."""

import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import numpy as np
from shapely.geometry import Polygon

from dynamic_functions.Terrain import arctic_dem, coastline, dataforsyningen, demand
from dynamic_functions.Terrain.acquisition_dates import date_range, normalize_date
from dynamic_functions.Terrain.Database import schema
from dynamic_functions.Terrain.Database.tiles import write_dem, read_dem_payload, TileClobberError
from dynamic_functions.Terrain.Database.textures import (
    write_texture_metatile, read_texture_payload, TextureClobberError,
)
from dynamic_functions.Terrain.binary_batch import compose_tiles_binary_from_ready_data
from dynamic_functions.Terrain.composition import compose_tiles_from_ready_data
from dynamic_functions.Terrain.dem_acquisition import fetch_best_dem


_TILE = "12-1978-92"
_SPOT = b'''<msGMLOutput><spot_optagetidspunkt_layer>
<spot_optagetidspunkt_feature><timeutc>2021-07-29 13:55:06</timeutc></spot_optagetidspunkt_feature>
<spot_optagetidspunkt_feature><timeutc>2019-08-01 12:00:00</timeutc></spot_optagetidspunkt_feature>
</spot_optagetidspunkt_layer></msGMLOutput>'''


def _dates(start="2010-06-04T00:00:00Z", end="2020-11-09T00:00:00Z"):
    return date_range([start, end], source="fixture", scope="fixture")


class AcquisitionDatesTest(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        schema.create(self.connection)
        self.heightmap = np.full((65, 65), 20, dtype=np.float32)
        self.mask = np.zeros((65, 65), dtype=bool)
        self.children = {c["tileId"]: b"fixture" for c in dataforsyningen._metatile_spec(_TILE)["children"]}
        self.child_dates = {tile_id: _dates() for tile_id in self.children}

    def tearDown(self):
        self.connection.close()

    def test_provider_date_parsing(self):
        dates = dataforsyningen._parse_spot_dates(_SPOT)
        self.assertEqual(dates["date"], "2019-08-01T12:00:00Z")
        self.assertEqual(dates["dateEnd"], "2021-07-29T13:55:06Z")
        self.assertEqual(dates["dateScope"], "tile_center")
        self.assertEqual(normalize_date("2020-01-01T02:00:00+02:00"), "2020-01-01T00:00:00Z")
        self.assertIsNone(dataforsyningen._parse_spot_dates(b"<msGMLOutput/>")["date"])
        self.assertFalse(date_range([None, "2020-01-01"], source="x", scope="x")["dateComplete"])
        with self.assertRaises(ValueError):
            dataforsyningen._parse_spot_dates(_SPOT.replace(b"2021-07-29 13:55:06", b"invalid"))
        with self.assertRaises(ValueError):
            dataforsyningen._parse_spot_dates(b"<ServiceExceptionReport/>")

    def test_spot_queries_each_child_and_surfaces_failures(self):
        with patch.object(dataforsyningen, "_http_get", return_value=(_SPOT, {})) as fetch:
            dates = dataforsyningen._fetch_metatile_dates(dataforsyningen._request_spec(_TILE), "test")
        self.assertEqual(set(dates), set(self.children))
        queries = [parse_qs(urlsplit(call.args[0]).query) for call in fetch.call_args_list]
        self.assertEqual(len({query["BBOX"][0] for query in queries}), 16)
        self.assertTrue(all(query["QUERY_LAYERS"] == ["spot_optagetidspunkt"] for query in queries))
        with patch.object(dataforsyningen, "_http_get", return_value=(None, {"status": "network_error"})):
            with self.assertRaises(ConnectionError):
                dataforsyningen._fetch_child_dates(_TILE, "test")

    def test_arctic_stac_and_contributing_sources(self):
        arctic_dem._source_dates.cache_clear()
        url = "https://provider.invalid/12_37_10m_v4.1_dem.tif"
        item = {"assets": {"dem": {"href": url}}, "properties": {
            "start_datetime": "2010-06-04T00:00:00Z", "end_datetime": "2020-11-09T00:00:00Z",
        }}
        with patch.object(arctic_dem.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(item).encode())) as fetch:
            dates = arctic_dem._source_dates(url)
        self.assertEqual(dates["dateEnd"], item["properties"]["end_datetime"])
        self.assertTrue(fetch.call_args.args[0].endswith("12_37_10m_v4.1.json"))
        with (
            patch.object(arctic_dem, "_sources_for_bbox", return_value=[{"url": "first"}, {"url": "unused"}]),
            patch.object(arctic_dem, "_decode_source", return_value=self.heightmap.copy()),
            patch.object(arctic_dem, "_source_dates", return_value=_dates()) as metadata,
            patch.object(arctic_dem, "_correct_vertical_datum", side_effect=lambda h, b: (h, 28.0)),
        ):
            _, sources, _ = arctic_dem._fetch_heightmap(_TILE)
        self.assertEqual([source["url"] for source in sources], ["first"])
        metadata.assert_called_once_with("first")
        arctic_dem._source_dates.cache_clear()

    def test_selected_dem_dates_do_not_leak_to_copernicus(self):
        with (
            patch("dynamic_functions.Terrain.dem_acquisition._fetch_arcticdem", return_value=(self.heightmap, [{"url": "x", **_dates()}], 28.0)),
            patch("dynamic_functions.Terrain.dem_acquisition._fetch_copernicus", return_value=(self.heightmap, [])),
        ):
            result = fetch_best_dem(_TILE)
        self.assertEqual(result["acquisitionDates"]["date"], _dates()["date"])
        partial = self.heightmap.copy()
        partial[0] = np.nan
        with (
            patch("dynamic_functions.Terrain.dem_acquisition._fetch_arcticdem", return_value=(partial, [{"url": "x", **_dates()}], 28.0)),
            patch("dynamic_functions.Terrain.dem_acquisition._fetch_copernicus", return_value=(self.heightmap, [])),
        ):
            self.assertIsNone(fetch_best_dem(_TILE)["acquisitionDates"])

    def test_partial_stac_dates_remain_incomplete_after_selection(self):
        partial = _dates(None, "2020-11-09T00:00:00Z")
        with (
            patch("dynamic_functions.Terrain.dem_acquisition._fetch_arcticdem", return_value=(self.heightmap, [partial], 28.0)),
            patch("dynamic_functions.Terrain.dem_acquisition._fetch_copernicus", return_value=(self.heightmap, [])),
        ):
            result = fetch_best_dem(_TILE)
        self.assertFalse(result["acquisitionDates"]["dateComplete"])

    def test_gtk_dates_only_from_intersecting_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gpkg"
            source = sqlite3.connect(path)
            for table in ("tidalwater_s", "island_s"):
                source.execute(f"CREATE TABLE {table} (geom BLOB, spatialsourcedatetime TEXT)")
            for bounds, date in [((0, 0, 10, 10), "2021-01-01"), ((30, 30, 40, 40), "1990-01-01")]:
                blob = b"GP\x00\x01" + b"\x00" * 4 + Polygon.from_bounds(*bounds).wkb
                source.execute("INSERT INTO tidalwater_s VALUES (?, ?)", (blob, date))
            source.commit()
            source.close()
            dates = []
            with patch.object(coastline, "shapely_transform", side_effect=lambda transform, geometry: geometry):
                water, islands = coastline._read_block(path, bbox=(1, 1, 2, 2), dates=dates)
            self.assertEqual(dates, ["2021-01-01"])
            self.assertEqual(len(water), 1)
            self.assertEqual(islands, [])

    def _publish(self):
        write_dem(self.connection, _TILE, self.heightmap, "arcticdem_10m", "EGM2008", acquisition_dates=_dates())
        write_texture_metatile(self.connection, self.children, "fixture", acquisition_dates=self.child_dates)
        coastline.write_coastline_mask(self.connection, _TILE, self.mask, "fixture", 2, acquisition_dates=_dates("2016-01-01", "2018-01-01"))

    def test_persistence_ancestor_and_binary_response(self):
        self._publish()
        descendant = "13-3956-184"
        result = compose_tiles_from_ready_data(self.connection, [descendant])["tiles"][0]
        self.assertEqual(result["provenance"]["heightmapDate"], _dates()["date"])
        self.assertEqual(result["provenance"]["textureDateEnd"], _dates()["dateEnd"])
        self.assertEqual(result["provenance"]["coastlineDate"], "2016-01-01T00:00:00Z")
        self.assertTrue(all(result["provenance"][key] == _TILE for key in ("heightmapTileId", "textureTileId", "coastlineTileId")))
        body, header = compose_tiles_binary_from_ready_data(self.connection, [descendant])
        length = int.from_bytes(body[:4], "little")
        wire = json.loads(body[4:4+length])
        self.assertEqual(wire["tiles"][0]["provenance"], result["provenance"])
        _, reused = compose_tiles_binary_from_ready_data(self.connection, [descendant], {descendant: header["tiles"][0]["heightmap"]})
        self.assertEqual(reused["tiles"][0]["heightmapBytes"], 0)
        self.assertEqual(reused["tiles"][0]["provenance"], result["provenance"])

    def test_date_conflicts_cannot_clobber_payloads(self):
        self._publish()
        self.assertFalse(write_dem(self.connection, _TILE, self.heightmap, "arcticdem_10m", "EGM2008", acquisition_dates=_dates()))
        with self.assertRaises(TileClobberError):
            write_dem(self.connection, _TILE, self.heightmap, "arcticdem_10m", "EGM2008", acquisition_dates=_dates("2000-01-01", "2001-01-01"))
        changed = dict(self.child_dates)
        changed[_TILE] = _dates("2000-01-01", "2001-01-01")
        with self.assertRaises(TextureClobberError):
            write_texture_metatile(self.connection, self.children, "fixture", acquisition_dates=changed)
        with self.assertRaises(coastline.CoastlineClobberError):
            coastline.write_coastline_mask(self.connection, _TILE, self.mask, "fixture", 2, acquisition_dates=_dates())
        self.assertEqual(read_texture_payload(self.connection, _TILE)["acquisition_dates"], _dates())

    def test_worker_writes_dates_and_metadata_failure_writes_nothing(self):
        acquisition = {"heightmap": self.heightmap, "source": "arcticdem_10m", "verticalDatum": "EGM2008", "provider": "arcticdem", "sources": [], "geoidUndulation": 28.0, "attempts": [], "acquisitionDates": _dates()}
        with patch.object(demand, "db", return_value=self.connection), patch.object(demand, "fetch_best_dem", return_value=acquisition):
            demand._dem_worker(_TILE)
        with (
            patch.object(demand, "db", return_value=self.connection),
            patch.dict(os.environ, {"DATAFORSYNINGEN_TOKEN": "test"}),
            patch.object(demand, "_fetch_metatile", return_value=(b"fixture", {"childAcquisitionDates": self.child_dates})),
            patch.object(demand, "_split_metatile", return_value=self.children),
        ):
            demand._texture_worker(_TILE)
        with patch.object(demand, "db", return_value=self.connection), patch.object(demand, "_acquire_coastline", return_value=(self.mask, {"acquisitionDates": _dates()})):
            demand._coastline_worker(_TILE)
        for reader in (read_dem_payload, read_texture_payload, coastline.read_coastline_mask):
            self.assertEqual(reader(self.connection, _TILE)["acquisition_dates"], _dates())
        with (
            patch.dict(os.environ, {"DATAFORSYNINGEN_TOKEN": "test"}),
            patch.object(demand, "_fetch_metatile", side_effect=ConnectionError("metadata failed")),
            patch.object(demand, "db") as database,
        ):
            with self.assertRaises(ConnectionError):
                demand._texture_worker(_TILE)
            database.assert_not_called()

    def test_migration_preserves_legacy_data_and_unknown_dates(self):
        write_dem(self.connection, _TILE, self.heightmap, "copernicus", "EGM2008")
        for table in ("tiles", "textures", "coastline_masks"):
            self.connection.execute(f"ALTER TABLE {table} DROP COLUMN acquisition_dates")
        schema.create(self.connection)
        schema.create(self.connection)
        payload = read_dem_payload(self.connection, _TILE)
        np.testing.assert_array_equal(payload["heightmap"], self.heightmap)
        self.assertIsNone(payload["acquisition_dates"]["date"])


@visible
def acquisition_dates_offline() -> dict:
    """Verify provider dates, atomic persistence, and JSON/binary provenance."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AcquisitionDatesTest)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(output.getvalue())
    return {"passed": True, "testsRun": result.testsRun}
