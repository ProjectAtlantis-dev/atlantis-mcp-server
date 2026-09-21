import unittest
from unittest.mock import patch
import numpy as np
from dynamic_functions.Terrain.dem_seams import read_continuous_dem


class SharedDemTests(unittest.TestCase):
    def setUp(self):
        self.tiles = {}
        for key, value in [('2-1-1', 10), ('2-2-1', 12), ('2-1-2', 14), ('2-2-2', 16)]:
            self.tiles[key] = {'heightmap': np.full((5, 5), value, dtype=np.float32),
                               'vertical_datum': 'EGM2008', 'updated_at': 'fixture',
                               'confidence_map': np.ones((5, 5), dtype=np.uint8)}
        mocked = patch('dynamic_functions.Terrain.dem_seams.read_dem_payload',
                       side_effect=lambda db, key: self.tiles.get(key))
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_independent_requests_agree_on_edges_and_four_tile_corner(self):
        a = read_continuous_dem(None, '2-1-1')['heightmap']
        b = read_continuous_dem(None, '2-2-1')['heightmap']
        c = read_continuous_dem(None, '2-1-2')['heightmap']
        d = read_continuous_dem(None, '2-2-2')['heightmap']
        np.testing.assert_array_equal(a[:, -1], b[:, 0])
        np.testing.assert_array_equal(a[-1, :], c[0, :])
        self.assertEqual(a[-1, -1], 13)
        self.assertEqual(a[-1, -1], d[0, 0])
        self.assertTrue(np.all(a[1:-1, 1:-1] == 10))
        self.assertTrue(np.all(self.tiles['2-1-1']['heightmap'] == 10))

    def test_request_order_and_cache_do_not_change_results(self):
        cache = {}
        first = {key: read_continuous_dem(None, key, cache)['heightmap'] for key in self.tiles}
        cache = {}
        for key in reversed(self.tiles):
            np.testing.assert_array_equal(first[key], read_continuous_dem(None, key, cache)['heightmap'])

    def test_unknown_samples_are_not_filled(self):
        self.tiles['2-1-1']['heightmap'][2, -1] = np.nan
        self.assertTrue(np.isnan(read_continuous_dem(None, '2-1-1')['heightmap'][2, -1]))
        self.assertEqual(read_continuous_dem(None, '2-2-1')['heightmap'][2, 0], 12)

    def test_different_vertical_datum_is_not_averaged(self):
        self.tiles['2-2-1']['vertical_datum'] = 'ellipsoid'
        self.assertEqual(read_continuous_dem(None, '2-1-1')['heightmap'][2, -1], 10)
