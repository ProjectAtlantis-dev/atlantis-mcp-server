"""Callback shell preferences must reach Node without selecting a default tab locally."""
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import atlantis
from call_context import CallContext, ToolCallPayload


class ShellRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.token = atlantis._current_context.set(CallContext(
            client_id="client", request_id="request", client_log_func=AsyncMock(),
            payload=ToolCallPayload(
                caller_sid="alice", user_game_id=1, exec_shell_path="20",
                caller_shell_path="1", display_shell_path="3", user_shell_path="4",
            ),
        ))
        atlantis._request_seq_counters.clear()

    def tearDown(self):
        atlantis._current_context.reset(self.token)
        atlantis._request_seq_counters.clear()

    async def test_command_targets_reach_wire(self):
        with patch.object(atlantis, "execute_client_command_awaitable", new_callable=AsyncMock) as send:
            for target, expected in [
                ("d_map", "d_map"), ("t_main", "t_main"), ("u_roster", "u_roster"),
                ("display", "display"), ("user", "user"), ("terminal", "terminal"),
                ("7.2", "7.2"), ("unknown", "unknown"), ("", "caller"),
                ("exec", "20"), ("caller", "1"), (" d_map ", "d_map"),
            ]:
                with self.subTest(target=target):
                    await atlantis.client_command("test", shell=target)
                    self.assertEqual(send.call_args.kwargs["shell_path"], expected)

    async def test_log_targets_and_omitted_default(self):
        with patch.object(atlantis, "util_client_log", new_callable=AsyncMock, return_value=None) as send:
            for target in ["d_map", "display", "terminal", "unknown"]:
                await atlantis.client_log("hello", shell=target)
                self.assertEqual(send.call_args.kwargs["shell_path"], target)
            await atlantis.client_log("hello")
            self.assertIsNone(send.call_args.kwargs["shell_path"])

    async def test_html_and_widget_keep_named_target(self):
        with patch.object(atlantis, "execute_client_command_awaitable", new_callable=AsyncMock) as send:
            await atlantis.client_html("<p>Map</p>", shell="d_map")
            self.assertEqual(send.call_args.kwargs["shell_path"], "d_map")
            widget = await atlantis.client_widget("one", widget_key="map", shell="d_map")
            await widget.update("two")
            await widget.remove()
            self.assertTrue(all(call.kwargs["shell_path"] == "d_map" for call in send.call_args_list))
            self.assertEqual(send.call_args.kwargs["message_params"]["action"], "remove")

    async def test_types_do_not_require_default_paths_in_context(self):
        token = atlantis._current_context.set(CallContext(
            client_id="client", request_id="request",
            payload=ToolCallPayload(caller_sid="alice", user_game_id=1, caller_shell_path="1"),
        ))
        try:
            with patch.object(atlantis, "execute_client_command_awaitable", new_callable=AsyncMock) as send:
                await atlantis.client_command("test", shell="display")
                self.assertEqual(send.call_args.kwargs["shell_path"], "display")
            self.assertEqual(atlantis._resolve_callback_shell("exec"), "1")
            with patch.object(atlantis, "get_exec_shell_path", return_value=None):
                with self.assertRaises(RuntimeError):
                    atlantis._resolve_callback_shell("exec")
        finally:
            atlantis._current_context.reset(token)

    async def test_server_errors_propagate(self):
        with patch.object(atlantis, "execute_client_command_awaitable", new_callable=AsyncMock,
                          side_effect=RuntimeError("server failed")):
            with self.assertRaisesRegex(RuntimeError, "server failed"):
                await atlantis.client_command("test", shell="d_map")

    async def test_sequence_numbers_remain_per_target(self):
        with patch.object(atlantis, "execute_client_command_awaitable", new_callable=AsyncMock) as send:
            for target in ["d_map", "d_other", "d_map"]:
                await atlantis.client_command("test", shell=target)
            self.assertEqual([c.kwargs["seq_num"] for c in send.call_args_list], [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
