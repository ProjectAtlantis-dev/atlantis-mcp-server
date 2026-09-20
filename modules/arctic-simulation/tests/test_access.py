import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from starlette.applications import Starlette
from starlette.testclient import TestClient
from atlantis_host_adapters.identity import resolve_principal
from atlantis_simulation.viewer import ViewerCapabilities, routes


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'bindings.json'
        self.binding = dict(caller='test-caller', user_game_id=7, external_user_id='x_user:42',
                            scenario='isolated-proof', permissions=['simulation', 'bank', 'world'])
        self.write([self.binding])
        self.env = patch.dict(os.environ, ATLANTIS_IDENTITY_BINDINGS=str(self.path))
        self.env.start()
        self.addCleanup(self.env.stop)
        self.ctx = SimpleNamespace(caller_sid='test-caller', user_game_id=7)

    def write(self, bindings):
        self.path.write_text(json.dumps({'version': 1, 'bindings': bindings}))

    def test_scope_and_identity(self):
        self.assertEqual(resolve_principal(self.ctx, 'bank').external_user_id, 'x_user:42')
        for context in (None, SimpleNamespace(caller_sid='other', user_game_id=7),
                        SimpleNamespace(caller_sid='test-caller', user_game_id=8)):
            with self.assertRaises(PermissionError):
                resolve_principal(context, 'simulation')

    def test_no_sid_fallback_or_duplicate_binding(self):
        with patch.dict(os.environ, {'GAME_BANK_ALLOW_SID_IDENTITY': '1'}, clear=True):
            with self.assertRaises(PermissionError):
                resolve_principal(self.ctx, 'bank')
        self.write([self.binding, self.binding])
        with self.assertRaises(ValueError):
            resolve_principal(self.ctx, 'bank')

    def test_expiry_scope_and_revocation(self):
        now = [10]
        caps = ViewerCapabilities(clock=lambda: now[0])
        grant = caps.issue(resolve_principal(self.ctx, 'simulation'), ttl=20)
        caps.check(grant['token'], 'isolated-proof')
        with self.assertRaises(PermissionError):
            caps.check(grant['token'], 'another-game')
        now[0] = 30
        with self.assertRaises(PermissionError):
            caps.check(grant['token'], 'isolated-proof')
        now[0] = 10
        self.write([])
        with self.assertRaises(PermissionError):
            caps.check(grant['token'], 'isolated-proof')

    def test_viewer_is_read_only_and_child_token_not_exposed(self):
        caps = ViewerCapabilities()
        grant = caps.issue(resolve_principal(self.ctx, 'simulation'))
        client = TestClient(Starlette(routes=routes()))
        with patch('atlantis_simulation.viewer.capabilities', caps), patch('atlantis_simulation.viewer.simulation_host') as host:
            host.command.return_value = {'protocol': 'atlantis-simulation-v1', 'tick': 9}
            self.assertEqual(client.get(grant['snapshotUrl']).status_code, 403)
            headers = {'Authorization': 'Bearer ' + grant['token']}
            response = client.get(grant['snapshotUrl'], headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['cache-control'], 'no-store')
            host.command.assert_called_once_with('GET', '/games/isolated-proof/snapshot')
            self.assertEqual(client.post(grant['snapshotUrl'], headers=headers).status_code, 405)
            self.assertEqual(client.get('/api/simulation/other/snapshot', headers=headers).status_code, 403)
            host.command.side_effect = RuntimeError('private child configuration')
            response = client.get(grant['snapshotUrl'], headers=headers)
            self.assertEqual(response.status_code, 503)
            self.assertNotIn('private', response.text)
