"""Focused gateway tests. No live account, money or server changes."""
import unittest
from types import SimpleNamespace
from atlantis_economy.gateway import execute, request_key

ACCOUNT = "11111111-1111-4111-8111-111111111111"
RECIPIENT = "22222222-2222-4222-8222-222222222222"


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.principal = SimpleNamespace(external_user_id="x_user:42", caller="Alice",
                                         permissions=frozenset({"bank"}))
        self.calls = []

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        return {"id": ACCOUNT} if path == "/accounts/resolve" else {"ok": True}

    def test_transfer_derives_debit_and_actor_from_trusted_account(self):
        execute(self.principal, "transfer_credits", request=self.request,
                to_account_id=RECIPIENT, amount=10, currency="GLC", idempotency_key="pay-1")
        body = self.calls[-1][2]
        self.assertEqual(body["fromAccountId"], ACCOUNT)
        self.assertEqual(body["actorAccountId"], ACCOUNT)
        self.assertEqual(body["toAccountId"], RECIPIENT)

    def test_no_actor_override_or_mint_operation(self):
        for operation, args in [("mint", {}), ("portfolio", {"ownerAccountId": RECIPIENT})]:
            with self.assertRaises(ValueError):
                execute(self.principal, operation, request=self.request, **args)
        self.assertEqual(self.calls, [])

    def test_denied_principal_never_calls_bank(self):
        self.principal.permissions = frozenset({"simulation"})
        with self.assertRaises(PermissionError):
            execute(self.principal, "portfolio", request=self.request)
        self.assertEqual(self.calls, [])

    def test_keys_stable_between_chats_but_separate_between_players(self):
        first = request_key(self.principal, "transfer_credits", "pay-1")
        self.principal.user_game_id = 999
        self.assertEqual(first, request_key(self.principal, "transfer_credits", "pay-1"))
        self.principal.external_user_id = "x_user:43"
        self.assertNotEqual(first, request_key(self.principal, "transfer_credits", "pay-1"))

    def test_invalid_asset_or_money_rejected_before_network(self):
        with self.assertRaises(ValueError):
            execute(self.principal, "verify", request=self.request, asset_id="amv-01")
        with self.assertRaises(ValueError):
            execute(self.principal, "transfer_credits", request=self.request,
                    to_account_id=RECIPIENT, amount=float("nan"), currency="GLC", idempotency_key="x")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
