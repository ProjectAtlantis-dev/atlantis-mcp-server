import json,sqlite3,unittest
from unittest.mock import patch
import numpy as np
from dynamic_functions.Terrain.Asset.grounding import plan_building_grounding,apply_building_grounding

class BuildingGroundingTests(unittest.TestCase):
    def setUp(self):
        self.assets=sqlite3.connect(':memory:');self.terrain=sqlite3.connect(':memory:')
        self.addCleanup(self.assets.close);self.addCleanup(self.terrain.close)
        self.assets.execute('CREATE TABLE assets(id,type,enabled,cx,cy,z,properties,updated_at)')
        self.assets.execute('INSERT INTO assets VALUES(?,?,?,?,?,?,?,?)',('house','BYGNING',1,5,5,81,json.dumps({'groundZ':81,'ring':[[1,2,82],[3,4,82],[4,2,82]]}),'old'))
        self.assets.commit()
        self.terrain.execute('CREATE TABLE tiles(tile_id,x_min,y_min,x_max,y_max,depth,heightmap,confidence_map)')
        self.terrain.execute("INSERT INTO tiles VALUES('tile',0,0,10,10,12,1,1)")
        self.payload={'heightmap':np.array([[50.,52.],[54.,56.]]),'vertical_datum':'EGM2008','updated_at':'sample'}
    def test_saved_ground_uses_dem_not_roof_clamp_and_preserves_original(self):
        with patch('dynamic_functions.Terrain.Asset.grounding.read_continuous_dem',return_value=self.payload):
            plan,unresolved=plan_building_grounding(self.assets,self.terrain)
            self.assertEqual(unresolved,[]);self.assertEqual(plan[0]['groundZ'],53)
            self.assertEqual(apply_building_grounding(self.assets,plan),1)
            z,raw,x,y=self.assets.execute('SELECT z,properties,cx,cy FROM assets').fetchone();p=json.loads(raw)
            self.assertEqual((z,x,y),(53,5,5));self.assertEqual(p['originalGroundZ'],81);self.assertEqual(p['ring'][0],[1,2,82])
            self.assertEqual(plan_building_grounding(self.assets,self.terrain),([],[]))
    def test_unknown_sample_is_reported_and_never_replaced_by_roof(self):
        self.payload['heightmap'][0,0]=np.nan
        with patch('dynamic_functions.Terrain.Asset.grounding.read_continuous_dem',return_value=self.payload):
            plan,unknown=plan_building_grounding(self.assets,self.terrain)
        self.assertEqual(plan,[]);self.assertEqual(unknown[0]['reason'],'unverified DEM sample')
    def test_concurrent_edit_aborts_the_reconciliation(self):
        with patch('dynamic_functions.Terrain.Asset.grounding.read_continuous_dem',return_value=self.payload):plan,_=plan_building_grounding(self.assets,self.terrain)
        self.assets.execute("UPDATE assets SET properties='{}'");self.assets.commit()
        with self.assertRaisesRegex(RuntimeError,'changed during grounding'):apply_building_grounding(self.assets,plan)
