import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dynamic_functions.Terrain.Objects import runtime

ID = '12345678-1234-4234-8234-123456789abc'


class ObjectFunctionsTests(unittest.TestCase):
    def setUp(self):
        self.principal = SimpleNamespace(caller='Tester', scenario='world', permissions=['simulation'],
                                         external_user_id='subject', user_game_id=1)
        self.asset = {'id': ID, 'assetType': 'black-hornet', 'kind': 'vehicle',
                      'ownerAccountId': 'owner', 'metadata': {'world': 'world'}}
        self.state = {'id': ID, 'controlStatus': 'stopped', 'mission': None}
        self.snapshot = {'controlledVehicles': [self.state]}
        def bank(*args):
            return {'authentic': True, 'spendable': True, 'asset': self.asset}
        def execute(principal, operation, asset_id, **kwargs):
            if operation == 'capabilities':
                return {'actions': [{'id': 'fly_to'}], 'missionActions': ['pause', 'resume', 'cancel', 'complete_task']}
            return {'operation': operation, 'assetId': asset_id, **kwargs}
        for name, value in [('account_for', lambda _: {'id': 'owner'}), ('bank_request', bank),
                            ('_snapshot', lambda _: self.snapshot), ('vehicle_execute', execute)]:
            p = patch.object(runtime, name, value);p.start();self.addCleanup(p.stop)

    def actions(self):
        return {a['id']: a for a in runtime.describe(self.principal, ID)['actions']}

    def test_only_currently_runnable_actions(self):
        self.assertIn('fly_to', self.actions())
        self.state['mission'] = {'id': 'm', 'status': 'running'}
        self.assertEqual(set(self.actions()), {'inspect', 'mission_status', 'pause', 'cancel'})
        self.state['mission']['status'] = 'awaiting_task'
        self.assertIn('complete_task', self.actions())
        self.state['mission']['status'] = 'paused'
        self.assertIn('resume', self.actions());self.assertNotIn('pause', self.actions())

    def test_no_movement_for_unattached_or_manual_control(self):
        self.state['controlled'] = True
        self.assertNotIn('fly_to', self.actions())
        self.snapshot['controlledVehicles'] = []
        self.assertEqual(set(self.actions()), {'inspect'})

    def test_ownership_is_checked_before_discovery_or_execution(self):
        self.asset['ownerAccountId'] = 'someone-else'
        with self.assertRaises(PermissionError):self.actions()
        with self.assertRaises(PermissionError):runtime.invoke(self.principal, ID, 'inspect', {})

    def test_wrong_world_is_rejected(self):
        self.asset['metadata']['world'] = 'elsewhere'
        with self.assertRaises(PermissionError):self.actions()

    def test_coordinate_fields_use_the_shared_vehicle_gateway(self):
        result = runtime.invoke(self.principal, ID, 'fly_to', {'latitude': 64, 'longitude': -51,
            'request_id': 'test', 'return_latitude': 63, 'return_longitude': -52,
            'wait_for_task': True, 'return_land': True})
        self.assertEqual(result['operation'], 'fly_to')
        self.assertEqual(result['parameters']['destination'], {'lat': 64, 'lon': -51})
        self.assertEqual(result['parameters']['returnDestination'], {'lat': 63, 'lon': -52, 'altitudeAglM': 60, 'landing': True})
        self.assertTrue(result['parameters']['waitForTask'])

    def test_partial_return_and_unknown_parameters_reject(self):
        with self.assertRaises(ValueError):runtime.invoke(self.principal, ID, 'fly_to', {'latitude': 64, 'longitude': -51, 'request_id': 't', 'return_latitude': 63})
        with self.assertRaises(ValueError):runtime.invoke(self.principal, ID, 'inspect', {'ownerAccountId': 'other'})
        with self.assertRaises(ValueError):runtime.invoke(self.principal, ID, 'arbitrary_tool', {})

    def test_stale_mission_form_cannot_act_on_new_mission(self):
        self.state['mission'] = {'id': 'new', 'status': 'running'}
        with self.assertRaises(ValueError):runtime.invoke(self.principal, ID, 'cancel', {'mission_id': 'old', 'action': 'cancel'})
        self.assertEqual(runtime.invoke(self.principal, ID, 'cancel', {'mission_id': 'new', 'action': 'cancel'})['parameters']['missionId'], 'new')

    def test_component_functions_respect_current_interlocks(self):
        state = {'outer': 1, 'inner': 0, 'target': {'outer': 1, 'inner': 0}}
        self.assertFalse(runtime.component_action_available(state, 'airlock_open_inner'))
        self.assertTrue(runtime.component_action_available(state, 'airlock_close'))
        self.assertFalse(runtime.component_action_available({'entry': state, 'freight': 0, 'freightTarget': 0}, 'facility_freight_open'))


if __name__ == '__main__':
    unittest.main()
