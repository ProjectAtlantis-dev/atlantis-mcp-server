import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dynamic_functions.Terrain.Placement import gateway

class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.principal = SimpleNamespace(caller='Tester', permissions=['simulation','bank'], scenario='world')
        self.model = {'id':'test-model','upAxis':'Y','bounds':{'min':[0,-2,0]}}

    def test_model_uses_verified_ground_and_authored_bottom_then_bank_gateway(self):
        def command(method,path):
            return {'models':[self.model]} if path.endswith('catalog') else {'origin':{'latitude':64,'longitude':-51,'altitudeM':10}}
        with patch.object(gateway.atlantis,'get_owner_usernames',return_value=['Tester']),patch.object(gateway.simulation_host,'command',side_effect=command),patch.object(gateway.terrain_adapter,'configuration',return_value={}),patch.object(gateway.terrain_adapter,'elevation_grid',return_value={'heights':[25]*9}),patch.object(gateway.infrastructure_control,'place_for') as place:
            gateway.place_model(self.principal,'test-model','instance',64,-51,90,1)
            place.assert_called_once_with(self.principal,'test-model',{'x':0,'y':0,'z':18},90,'instance')
            with self.assertRaises(ValueError):gateway.place_model(self.principal,'test-model','instance',float('nan'),-51)
            self.principal.caller='Other'
            with self.assertRaises(PermissionError):gateway.place_model(self.principal,'test-model','instance',64,-51)
            self.assertEqual(place.call_count,1)

    def test_two_sites_issue_distinct_bank_assets_and_retries_reuse_them(self):
        from uuid import uuid4
        issued = {}
        sites = []
        def bank(method, path, body):
            key = (body['sourceNamespace'], body['sourceId'])
            if key not in issued:
                issued[key] = {**body, 'id': str(uuid4())}
            return issued[key]
        def command(method, path, body=None):
            if method == 'GET':
                return {'sites': sites}
            existing = next((site for site in sites if site['id'] == body['id']), None)
            if existing is None:
                sites.append(body)
            return existing or body
        with patch.object(gateway.atlantis,'get_owner_usernames',return_value=['Tester']),patch.object(gateway,'ground_position',return_value={'x':1,'y':2,'z':3}),patch.object(gateway.simulation_host,'command',side_effect=command),patch.object(gateway,'account_for',return_value={'id':'owner'}),patch.object(gateway,'bank_request',side_effect=bank):
            first = gateway.deploy_demo_site(self.principal,'one','point-defense',64,-51)
            second = gateway.deploy_demo_site(self.principal,'two','point-defense',64,-51)
            retry = gateway.deploy_demo_site(self.principal,'one','point-defense',64,-51)
            self.assertEqual(first, retry)
            self.assertEqual(len(issued), 10)
            self.assertEqual(len({a['id'] for a in issued.values()}), 10)
            self.assertEqual(len(first['bankAssets']), 5)
            self.assertNotEqual(first['bankAssets']['radar']['id'],second['bankAssets']['radar']['id'])
            with self.assertRaises(ValueError):gateway.deploy_demo_site(self.principal,'one','point-defense',65,-51)
            with self.assertRaises(ValueError):gateway.deploy_demo_site(self.principal,'one','microwave',64,-51)
            self.assertEqual(len(issued),10)

    def test_partial_bank_failure_retries_without_reissuing_completed_assets(self):
        from uuid import uuid4
        issued = {}
        fail = [True]
        def bank(method, path, body):
            key = body['sourceId']
            if key.endswith(':resupply') and fail[0]:
                raise RuntimeError('bank unavailable')
            issued.setdefault(key, {**body, 'id':str(uuid4())})
            return issued[key]
        with patch.object(gateway.atlantis,'get_owner_usernames',return_value=['Tester']),patch.object(gateway,'ground_position',return_value={'x':1,'y':2,'z':3}),patch.object(gateway.simulation_host,'command',return_value={'sites':[]}) as command,patch.object(gateway,'account_for',return_value={'id':'owner'}),patch.object(gateway,'bank_request',side_effect=bank):
            with self.assertRaisesRegex(RuntimeError,'bank unavailable'):
                gateway.deploy_demo_site(self.principal,'one','point-defense',64,-51)
            before = {key:asset['id'] for key,asset in issued.items()}
            self.assertTrue(all(call.args[0]=='GET' for call in command.call_args_list))
            fail[0] = False
            gateway.deploy_demo_site(self.principal,'one','point-defense',64,-51)
            self.assertEqual(len(issued),5)
            self.assertTrue(all(issued[key]['id']==value for key,value in before.items()))

    def test_mobile_placement_attaches_same_bank_uuid_including_retry(self):
        with patch.object(gateway,'assets',return_value={'models':[self.model]}), \
             patch.object(gateway,'ground_position',return_value={'x':0,'y':0,'z':20}), \
             patch.object(gateway.infrastructure_control,'place_for',return_value={'bankAssetId':'issued-id','alreadyPlaced':True}), \
             patch.object(gateway.equipment_control,'mobility_models',return_value={'test-model':{'domain':'ground'}}), \
             patch.object(gateway.equipment_control,'attach_movement',return_value={'attached':True}) as attach:
            result=gateway.place_model(self.principal,'test-model','same-key',64,-51)
        attach.assert_called_once_with(self.principal,'issued-id')
        self.assertEqual(result['bankAssetId'],'issued-id')
        self.assertTrue(result['movement']['attached'])

    def test_movement_attachment_failure_is_not_reported_as_success(self):
        with patch.object(gateway,'assets',return_value={'models':[self.model]}), \
             patch.object(gateway,'ground_position',return_value={'x':0,'y':0,'z':20}), \
             patch.object(gateway.infrastructure_control,'place_for',return_value={'bankAssetId':'issued-id'}), \
             patch.object(gateway.equipment_control,'mobility_models',return_value={'test-model':{'domain':'ground'}}), \
             patch.object(gateway.equipment_control,'attach_movement',side_effect=ValueError('terrain unavailable')):
            with self.assertRaisesRegex(ValueError,'terrain unavailable'):
                gateway.place_model(self.principal,'test-model','same-key',64,-51)
