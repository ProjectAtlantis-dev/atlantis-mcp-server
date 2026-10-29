"""Node UUID is the only game identity; only MCP owners create missing records."""
import ast
import builtins
import logging
import importlib
import sys
import tempfile
import unittest
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import atlantis
from call_context import CallContext

# The dynamic loader normally supplies these decorators.
with patch.object(builtins, 'public', lambda f: f, create=True), \
     patch.object(builtins, 'visible', lambda f: f, create=True):
    game = importlib.import_module('dynamic_functions.Chat.game')
    common = importlib.import_module('dynamic_functions.Chat.common')

GAME_UUID = '550e8400-e29b-41d4-a716-446655440000'
OTHER_UUID = '550e8400-e29b-41d4-a716-446655440001'
GAME = 'Setup'
OTHER_GAME = 'Greenland'


class GameIdentityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        for name in [GAME, OTHER_GAME]:
            (Path(temp.name) / name / 'Game').mkdir(parents=True)
        root_patch = patch.object(common, 'CHAT_HOME', temp.name)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        owner_patch = patch.object(atlantis, '_owner_usernames', ['owner'])
        owner_patch.start()
        self.addCleanup(owner_patch.stop)
        async def initialize(game_name, game_key):
            meta = game._game_read(GAME, game_key)
            meta['roster_scene'] = 'test_scene'
            game._game_update(GAME, game_key, meta)
            (Path(game.game_dir(GAME, game_key)) / 'roster.json').write_text('[]')

        self.game_init = AsyncMock(side_effect=initialize)
        runner = types.ModuleType('dynamic_functions.Chat.runner')
        runner.game_init = self.game_init
        for mocked in [
            patch.dict(sys.modules, {'dynamic_functions.Chat.runner': runner}),
            patch.object(atlantis, 'client_command', new_callable=AsyncMock),
        ]:
            mocked.start()
            self.addCleanup(mocked.stop)
        self.context = CallContext.from_params({
            'game_uuid': GAME_UUID, 'caller_sid': 'owner',
            'user_game_id': 6, 'caller_shell_path': '1',
        }, 'client', 'request')
        token = atlantis._current_context.set(self.context)
        self.addCleanup(atlantis._current_context.reset, token)

    def set_context(self, **updates):
        atlantis._current_context.set(self.context.with_payload_updates(**updates))

    async def test_owner_creates_once_using_node_uuid(self):
        self.assertEqual(await game.game_find_current(GAME), GAME_UUID)
        meta = game._game_read(GAME, GAME_UUID)
        self.assertEqual(meta['game_uuid'], GAME_UUID)
        self.assertEqual(meta['state'], 'stopped')
        self.assertIn('owner:' + GAME_UUID, meta['members'])
        self.assertEqual(await game.game_find_current(GAME), GAME_UUID)
        self.assertEqual(game._game_read(GAME, GAME_UUID), meta)

    async def test_creation_initializes_once_after_record_and_cursor_exist(self):
        async def initialize(game_name, game_key):
            self.assertEqual(game._game_read(GAME, game_key)['owner'], 'owner')
            atlantis.client_command.assert_awaited_once_with(
                '/cursor join', {'game_name': GAME, 'game_key': game_key,
                                 'join_password': game._game_read(GAME, game_key)['join_password']})
            meta = game._game_read(GAME, game_key)
            meta['roster_scene'] = 'test_scene'
            game._game_update(GAME, game_key, meta)
            (Path(game.game_dir(GAME, game_key)) / 'roster.json').write_text('[]')
        self.game_init.side_effect = initialize
        await game.game_find_current(GAME)
        await game.game_find_current(GAME)
        await game.game_new(GAME)
        self.game_init.assert_awaited_once_with(GAME, GAME_UUID)

    async def test_explicit_game_new_finishes_existing_empty_game(self):
        await game.game_new(GAME)
        meta = game._game_read(GAME, GAME_UUID)
        meta['roster_scene'] = None
        game._game_update(GAME, GAME_UUID, meta)
        (Path(game.game_dir(GAME, GAME_UUID)) / 'roster.json').unlink()
        self.game_init.reset_mock()
        await game.game_new(GAME)
        self.game_init.assert_awaited_once_with(GAME, GAME_UUID)

    async def test_setup_cancellation_propagates(self):
        self.game_init.side_effect = RuntimeError('Scene selection cancelled')
        with self.assertRaisesRegex(RuntimeError, 'Scene selection cancelled'):
            await game.game_new(GAME)
        self.assertEqual(game._game_read(GAME, GAME_UUID)['state'], 'stopped')

    async def test_existing_game_ignores_numeric_id_and_membership_for_lookup(self):
        await game.game_find_current(GAME)
        self.set_context(caller_sid='visitor', user_game_id=999)
        self.assertEqual(await game.game_find_current(GAME), GAME_UUID)
        # Lookup is not a membership grant.
        with self.assertRaises(PermissionError):
            game.require_membership(GAME, GAME_UUID)

    async def test_game_new_and_callbacks_share_the_same_record(self):
        keys = await game.game_new(GAME)
        self.assertEqual(keys['game_key'], GAME_UUID)
        await self.callback('preflight_callback')(GAME)
        self.assertEqual(await game.game_new(GAME), keys)
        self.assertEqual(await game.game_find_current(GAME), GAME_UUID)

    async def test_game_new_rejects_non_owner_even_for_existing_game(self):
        await game.game_new(GAME)
        self.set_context(caller_sid='visitor')
        with self.assertRaises(PermissionError):
            await game.game_new(GAME)

    async def test_non_owner_cannot_create(self):
        self.set_context(caller_sid='visitor')
        with self.assertRaises(PermissionError):
            await game.game_find_current(GAME)
        self.assertFalse(Path(game.game_dir(GAME, GAME_UUID)).exists())

    async def test_no_numeric_id_needed(self):
        self.set_context(user_game_id=None)
        self.assertEqual(await game.game_find_current(GAME), GAME_UUID)

    async def test_same_numeric_id_different_uuid_does_not_match(self):
        await game.game_find_current(GAME)
        self.set_context(game_uuid=OTHER_UUID, caller_sid='visitor')
        with self.assertRaises(PermissionError):
            await game.game_find_current(GAME)

    async def test_missing_or_malformed_uuid_never_falls_back(self):
        await game.game_find_current(GAME)
        for key in [None, '', '../escape', 'not-a-uuid']:
            with self.subTest(key=key):
                self.set_context(game_uuid=key)
                with self.assertRaises((RuntimeError, ValueError)):
                    await game.game_find_current(GAME)

    async def test_partial_record_is_not_recreated(self):
        Path(game.game_dir(GAME, GAME_UUID)).mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, 'Invalid game record'):
            await game.game_find_current(GAME)
        self.assertFalse((Path(game.game_dir(GAME, GAME_UUID)) / 'game.json').exists())

    async def test_corrupt_json_propagates(self):
        await game.game_find_current(GAME)
        (Path(game.game_dir(GAME, GAME_UUID)) / 'game.json').write_text('{broken')
        with self.assertRaises(ValueError):
            await game.game_find_current(GAME)

    def callback(self, name, speaker="visitor"):
        source = Path(game.__file__).with_name('chat_callback.py')
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
        node.decorator_list = []
        scope = {
            'atlantis': atlantis,
            'game_find_current': game.game_find_current,
            'game_require_current': game.game_require_current,
            'require_membership': game.require_membership,
            '_game_is_running': game._game_is_running,
            '_game_read': game._game_read,
            '_game_set_state': game._game_set_state,
            'GAME_STATE_RUNNING': game.GAME_STATE_RUNNING,
            'fetch_transcript': AsyncMock(return_value=([{'sid': speaker}], ['message'])),
            'analyze_participants': lambda rows: {'last_speaker': rows[-1]['sid']},
            '_handle_chat': AsyncMock(),
            '_BUSY_KEY': 'test_game_identity_busy',
            'logger': logging.getLogger(__name__),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        return scope[name]

    async def test_preflight_creates_game_before_chat_and_chat_keeps_it_stopped(self):
        await self.callback('preflight_callback')(GAME)
        meta = game._game_read(GAME, GAME_UUID)
        with self.assertRaisesRegex(PermissionError, 'is stopped'):
            await self.callback('chat_callback')(GAME)
        self.assertEqual(game._game_read(GAME, GAME_UUID), meta)

    async def test_chat_alone_never_creates_a_game(self):
        with self.assertRaisesRegex(RuntimeError, 'preflight has not set it up'):
            await self.callback('chat_callback', speaker='owner')(GAME)
        self.assertFalse(Path(game.game_dir(GAME, GAME_UUID)).exists())
        self.game_init.assert_not_awaited()

    async def test_owner_speech_resumes_and_processes_same_transcript(self):
        await game.game_new(GAME)
        callback = self.callback('chat_callback', speaker='owner')
        with patch.object(atlantis, 'client_log', new_callable=AsyncMock), \
             patch.object(atlantis, 'client_command', new_callable=AsyncMock) as command:
            await callback(GAME)
        self.assertEqual(game._game_read(GAME, GAME_UUID)['state'], 'running')
        callback.__globals__['_handle_chat'].assert_awaited_once_with(
            GAME, GAME_UUID, [{'sid': 'owner'}], ['message'])
        callback.__globals__['fetch_transcript'].assert_awaited_once()
        command.assert_not_awaited()

    async def test_owner_callback_context_does_not_resume_for_visitor_speech(self):
        await game.game_new(GAME)
        callback = self.callback('chat_callback', speaker='visitor')
        with self.assertRaisesRegex(PermissionError, 'only its owner may restart'):
            await callback(GAME)
        self.assertIsNone(atlantis.session_shared.get('test_game_identity_busy'))
        self.assertEqual(game._game_read(GAME, GAME_UUID)['state'], 'stopped')
        callback.__globals__['_handle_chat'].assert_not_awaited()

    def strip_roster(self):
        meta = game._game_read(GAME, GAME_UUID)
        meta['roster_scene'] = None
        game._game_update(GAME, GAME_UUID, meta)
        (Path(game.game_dir(GAME, GAME_UUID)) / 'roster.json').unlink()

    async def test_half_set_up_game_stays_stopped_and_chat_does_not_finish_it(self):
        await game.game_new(GAME)
        self.strip_roster()
        self.game_init.reset_mock()
        callback = self.callback('chat_callback', speaker='owner')
        with self.assertRaisesRegex(RuntimeError, 'preflight has not finished'):
            await callback(GAME)
        self.game_init.assert_not_awaited()
        self.assertEqual(game._game_read(GAME, GAME_UUID)['state'], 'stopped')
        self.assertIsNone(atlantis.session_shared.get('test_game_identity_busy'))

    async def test_preflight_finishes_half_set_up_game(self):
        await game.game_new(GAME)
        self.strip_roster()
        self.game_init.reset_mock()
        await self.callback('preflight_callback')(GAME)
        self.game_init.assert_awaited_once_with(GAME, GAME_UUID)
        self.assertEqual(game.game_require_current(GAME), GAME_UUID)

    async def test_non_owner_preflight_cannot_finish_half_set_up_game(self):
        await game.game_new(GAME)
        self.strip_roster()
        self.game_init.reset_mock()
        self.set_context(caller_sid='visitor')
        with self.assertRaises(PermissionError):
            await self.callback('preflight_callback')(GAME)
        self.game_init.assert_not_awaited()

    async def test_running_game_processes_visitor_speech(self):
        await game.game_new(GAME)
        meta = game._game_read(GAME, GAME_UUID)
        meta['state'] = 'running'
        game._game_update(GAME, GAME_UUID, meta)
        callback = self.callback('chat_callback', speaker='visitor')
        await callback(GAME)
        callback.__globals__['_handle_chat'].assert_awaited_once()

    async def test_callbacks_reject_missing_uuid_and_unauthorized_creation(self):
        # Chat refuses a missing game before ownership matters; preflight checks the owner.
        for name, missing_game_error in [('preflight_callback', PermissionError),
                                         ('chat_callback', RuntimeError)]:
            with self.subTest(callback=name):
                self.set_context(game_uuid=None)
                with self.assertRaises(RuntimeError):
                    await self.callback(name)(GAME)
                self.set_context(caller_sid='visitor')
                with self.assertRaises(missing_game_error):
                    await self.callback(name)(GAME)
        self.assertFalse(Path(game.game_dir(GAME, GAME_UUID)).exists())

    async def test_game_names_keep_separate_records_for_one_uuid(self):
        await game.game_new(GAME)
        self.assertFalse(Path(game.game_dir(OTHER_GAME, GAME_UUID)).exists())
        self.assertEqual(await game.game_find_current(OTHER_GAME), GAME_UUID)
        self.assertNotEqual(
            game._game_read(GAME, GAME_UUID)['join_password'],
            game._game_read(OTHER_GAME, GAME_UUID)['join_password'],
        )
        self.assertEqual(Path(game.game_dir(GAME, GAME_UUID)).parent.parent.parent.name, GAME)

    async def test_unknown_or_malformed_game_name_is_rejected(self):
        for name in ['Nope', '', '../Setup', 'Set up']:
            with self.subTest(game_name=name):
                with self.assertRaises(ValueError):
                    await game.game_find_current(name)

    def test_context_preserves_uuid_and_keys_sessions_by_uuid(self):
        context = self.context.with_payload_updates(user_game_id=999)
        self.assertEqual(context.game_uuid, GAME_UUID)
        self.assertEqual(context.session_key, 'owner:' + GAME_UUID)
        self.assertEqual(context.terminal_key, 'owner:' + GAME_UUID + ':1')
        self.assertIsNone(context.with_payload_updates(game_uuid=None).get_session_key())


if __name__ == '__main__':
    unittest.main()
