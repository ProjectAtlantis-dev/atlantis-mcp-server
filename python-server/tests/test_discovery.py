"""Every bot turn offers search, whatever the bot's per-game inventory holds."""
import builtins
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The dynamic loader normally supplies these decorators.
with patch.object(builtins, 'public', lambda f: f, create=True), \
     patch.object(builtins, 'visible', lambda f: f, create=True):
    turn = importlib.import_module('dynamic_functions.Chat.turn')


INVENTORY_TOOL = {
    'tool_name': 'remember_visitor',
    'tool_app': 'Chat',
    'remote_owner': 'owner',
    'remote_name': 'home',
    'tool_description': 'Remember a visitor',
    'input_schema': '{"type": "object", "properties": {}}',
}


def names(openai_tools):
    return [tool['function']['name'] for tool in openai_tools]


class DiscoveryTests(unittest.TestCase):
    def test_empty_inventory_still_offers_search(self):
        openai_tools, tool_lookup = turn.convert_search_tools([])
        turn._open_discovery(openai_tools, tool_lookup)
        self.assertEqual(names(openai_tools), ['search'])
        self.assertEqual(
            openai_tools[0]['function']['parameters']['required'], ['query'])
        self.assertEqual(tool_lookup['search']['functionName'], 'search')

    def test_search_joins_inventory_tools(self):
        openai_tools, tool_lookup = turn.convert_search_tools([INVENTORY_TOOL])
        turn._open_discovery(openai_tools, tool_lookup)
        self.assertEqual(names(openai_tools), ['remember_visitor', 'search'])
        self.assertEqual(set(tool_lookup), {'remember_visitor', 'search'})

    def test_inventory_declaring_search_is_an_error(self):
        openai_tools, tool_lookup = turn.convert_search_tools(
            [dict(INVENTORY_TOOL, tool_name='search')])
        with self.assertRaisesRegex(ValueError, 'already declares'):
            turn._open_discovery(openai_tools, tool_lookup)

    def test_each_turn_gets_its_own_schema(self):
        first_tools, first_lookup = [], {}
        turn._open_discovery(first_tools, first_lookup)
        first_tools[0]['function']['parameters']['properties'].clear()
        second_tools, second_lookup = [], {}
        turn._open_discovery(second_tools, second_lookup)
        self.assertIn('query', second_tools[0]['function']['parameters']['properties'])

    def test_closing_discovery_withdraws_search_only(self):
        openai_tools, tool_lookup = turn.convert_search_tools([INVENTORY_TOOL])
        turn._open_discovery(openai_tools, tool_lookup)
        turn._close_discovery(openai_tools, tool_lookup)
        self.assertEqual(names(openai_tools), ['remember_visitor'])
        self.assertEqual(set(tool_lookup), {'remember_visitor'})


class PreloadQueryTests(unittest.TestCase):
    def test_last_four_spoken_lines_oldest_first(self):
        transcript = [
            {'role': 'user', 'content': 'one'},
            {'role': 'assistant', 'content': 'two'},
            {'role': 'user', 'content': '[scene: the lobby]'},
            {'role': 'user', 'content': 'three'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'x'}]},
            {'role': 'tool', 'tool_call_id': 'x', 'content': '{"ok": true}'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'four\n  lines'}]},
            {'role': 'user', 'content': 'yes'},
        ]
        self.assertEqual(
            turn._recent_spoken_text(transcript, turn._PRELOAD_SEARCH_LINES),
            'two three four lines yes')

    def test_short_or_silent_transcript(self):
        self.assertEqual(turn._recent_spoken_text([{'role': 'user', 'content': 'hi'}], 4), 'hi')
        self.assertEqual(turn._recent_spoken_text([{'role': 'user', 'content': '[scene: x]'}], 4), '')


if __name__ == '__main__':
    unittest.main()
