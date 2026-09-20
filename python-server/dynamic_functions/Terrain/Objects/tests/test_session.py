import asyncio
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from atlantis_simulation.viewer import ViewerCapabilities
from dynamic_functions.Terrain.Objects import session


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.principal = SimpleNamespace(caller='Tester', scenario='world', user_game_id=1)
        self.capabilities = ViewerCapabilities(clock=lambda: self.now)
        self.grant = self.capabilities.issue(self.principal, 10, control_vehicle_id='owned')
        self.digest = hashlib.sha256(self.grant['token'].encode()).digest()
        self.request = SimpleNamespace(headers={'authorization': 'Bearer '+self.grant['token']}, path_params={'game_id': 'world'})
        for target, kwargs in [('dynamic_functions.Terrain.Objects.session.capabilities', {'new':self.capabilities}),
                               ('atlantis_simulation.viewer.resolve_principal', {'return_value':self.principal}),
                               ('dynamic_functions.Terrain.Objects.session.owned_asset', {'return_value':{}})]:
            p=patch(target, **kwargs);p.start();self.addCleanup(p.stop)

    def test_valid_session_renews_same_scope_and_token(self):
        self.now = 9
        response = asyncio.run(session.keep_alive(self.request))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.capabilities.grants[self.digest], (self.principal, 909, 'owned', None))

    def test_expired_session_cannot_be_revived(self):
        self.now = 11
        self.assertEqual(asyncio.run(session.keep_alive(self.request)).status_code,403)
        self.assertEqual(self.capabilities.grants[self.digest][1],10)

    def test_lost_ownership_cannot_extend_session(self):
        with patch.object(session,'owned_asset',side_effect=PermissionError('not owner')):
            self.assertEqual(asyncio.run(session.keep_alive(self.request)).status_code,403)
        self.assertEqual(self.capabilities.grants[self.digest][1],10)

    def test_anonymous_request_rejects(self):
        self.request.headers={}
        self.assertEqual(asyncio.run(session.keep_alive(self.request)).status_code,403)
