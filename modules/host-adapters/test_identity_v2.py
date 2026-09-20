import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from atlantis_host_adapters.identity import resolve_principal


class IdentityTests(unittest.TestCase):
    def test_persistent_subject_survives_chat_change_without_guessing_cloud_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_text(json.dumps({"version": 2, "bindings": [{"caller": "Alice",
                "subject_uuid": "11111111-1111-4111-8111-111111111111",
                "scenario": "greenland", "permissions": ["bank", "simulation"]}]}))
            first = resolve_principal(SimpleNamespace(caller_sid="Alice", user_game_id=1), "bank", policy_path=path)
            second = resolve_principal(SimpleNamespace(caller_sid="Alice", user_game_id=2), "bank", policy_path=path)
            self.assertEqual(first.external_user_id, second.external_user_id)
            self.assertEqual(first.scenario, second.scenario)
            with self.assertRaises(PermissionError):
                resolve_principal(SimpleNamespace(caller_sid="Mallory", user_game_id=1), "bank", policy_path=path)
            with self.assertRaises(PermissionError):
                resolve_principal(SimpleNamespace(caller_sid="Alice", user_game_id=1), "world", policy_path=path)


if __name__ == "__main__":
    unittest.main()
