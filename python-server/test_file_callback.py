"""Run with python3 -B test_file_callback.py; no live MCP server required."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).parent / "dynamic_functions" / "Home" / "file.py"


class FileCallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "files"
        self.root.mkdir()
        cwd = patch.object(Path, "cwd", side_effect=AssertionError("File storage must not use the process working directory"))
        cwd.start()
        self.addCleanup(cwd.stop)
        self.scope = {"__file__": str(self.root / "file.py"), "file": lambda function: function}
        exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), self.scope)

    def call(self, operation, filename, content=None):
        return asyncio.run(self.scope["file_callback"](operation, filename, content))

    def test_missing_file_raises_and_set_creates_file(self):
        with self.assertRaises(FileNotFoundError) as error:
            self.call("get", "sketch.excalidraw")
        self.assertEqual(error.exception.filename, str(self.root / "sketch.excalidraw"))
        self.assertEqual(list(self.root.iterdir()), [])
        scene = '{"type":"excalidraw","elements":[],"label":"猫"}'
        self.call("set", "sketch.excalidraw", scene)
        self.assertEqual((self.root / "sketch.excalidraw").read_text(), scene)
        self.assertEqual(self.call("get", "sketch.excalidraw"), scene)
        self.call("set", "sketch.excalidraw", "")
        self.assertEqual(self.call("get", "sketch.excalidraw"), "")

    def test_paths_are_rejected(self):
        for filename in [
            "", " ", ".", "..", "../outside", "/tmp/outside",
            "nested/../../outside", "nested/file", "./file", "file/",
            "nested\\file", "..\\outside", "C:\\outside", "C:outside",
            "\\\\server\\share\\file",
        ]:
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    self.call("get", filename)
                with self.assertRaises(ValueError):
                    self.call("set", filename, "data")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_symlink_escape_is_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.write_text("unchanged")
        (self.root / "link").symlink_to(outside)
        for operation in ["get", "set"]:
            with self.assertRaises(ValueError):
                self.call(operation, "link", "changed")
        self.assertEqual(outside.read_text(), "unchanged")

    def test_invalid_operation_and_missing_content(self):
        with self.assertRaises(ValueError):
            self.call("delete", "sketch")
        with self.assertRaises(ValueError):
            self.call("set", "sketch")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_io_errors_propagate(self):
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                self.call("get", "sketch")
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.call("set", "sketch", "data")

    def test_non_drawing_contents_are_read_unchanged(self):
        self.call("set", "example.py", "def example(): pass")
        self.assertEqual(self.call("get", "example.py"), "def example(): pass")

    def test_list_returns_only_script_folder_json_files(self):
        self.assertEqual(asyncio.run(self.scope["file_callback"]("list", suffix="json")), [])
        for filename in ["z.json", "a.json", "notes.txt"]:
            self.call("set", filename, "{}")
        nested = self.root / "nested.json"
        nested.mkdir()
        (nested / "hidden.json").write_text("{}")
        (self.root / "link.json").symlink_to(self.root / "a.json")
        self.assertEqual(
            asyncio.run(self.scope["file_callback"]("list", suffix="json")),
            [{"name": "a", "suffix": "json"}, {"name": "z", "suffix": "json"}],
        )
        self.assertEqual(
            asyncio.run(self.scope["file_callback"]("list", suffix="txt")),
            [{"name": "notes", "suffix": "txt"}],
        )

    def test_list_requires_a_valid_suffix(self):
        with self.assertRaises(ValueError):
            asyncio.run(self.scope["file_callback"]("list"))
        for suffix in ["", " ", ".json", "../json", "nested/json", "nested\\json"]:
            with self.subTest(suffix=suffix):
                with self.assertRaises(ValueError):
                    asyncio.run(self.scope["file_callback"]("list", suffix=suffix))

    def test_list_accepts_positional_suffix(self):
        self.call("set", "a.json", "{}")
        self.call("set", "notes.txt", "notes")
        self.assertEqual(self.call("list", "json"), [{"name": "a", "suffix": "json"}])
        for suffix in ["", " ", ".json", "../json", "nested/json", "nested\\json"]:
            with self.subTest(suffix=suffix):
                with self.assertRaises(ValueError):
                    self.call("list", suffix)

    def test_list_rejects_duplicate_suffix_arguments(self):
        for suffix in ["json", "txt"]:
            with self.subTest(suffix=suffix):
                with self.assertRaisesRegex(ValueError, "not both"):
                    asyncio.run(self.scope["file_callback"]("list", "json", suffix=suffix))

    def test_filename_extension_must_match_supplied_suffix(self):
        target = self.root / "sketch.txt"
        target.write_text("unchanged")
        for operation in ["get", "set"]:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, "does not match suffix"):
                    asyncio.run(self.scope["file_callback"](
                        operation, "sketch.txt", content="changed", suffix="json"))
        self.assertEqual(target.read_text(), "unchanged")
        self.assertEqual(list(self.root.iterdir()), [target])

    def test_matching_suffix_preserves_filename(self):
        asyncio.run(self.scope["file_callback"](
            "set", "sketch.json", content="{}", suffix="json"))
        self.assertEqual(asyncio.run(self.scope["file_callback"](
            "get", "sketch.json", suffix="json")), "{}")
        self.assertEqual([path.name for path in self.root.iterdir()], ["sketch.json"])

    def test_missing_extension_gets_suffix_appended(self):
        self.assertEqual(asyncio.run(self.scope["file_callback"](
            "set", "sketch", content="{}", suffix="json")), "Saved sketch.json")
        self.assertEqual([path.name for path in self.root.iterdir()], ["sketch.json"])
        self.assertEqual(asyncio.run(self.scope["file_callback"](
            "get", "sketch", suffix="json")), "{}")

    def test_get_and_set_reject_invalid_suffix(self):
        for operation in ["get", "set"]:
            for suffix in ["", ".json", "a/b"]:
                with self.subTest(operation=operation, suffix=suffix):
                    with self.assertRaisesRegex(ValueError, "Suffix must be"):
                        asyncio.run(self.scope["file_callback"](
                            operation, "sketch", content="{}", suffix=suffix))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_get_and_set_with_suffix_still_require_filename(self):
        for operation in ["get", "set"]:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, "Must specify a filename"):
                    asyncio.run(self.scope["file_callback"](operation, content="{}", suffix="json"))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_get_and_set_still_require_filename(self):
        for operation in ["get", "set"]:
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    asyncio.run(self.scope["file_callback"](operation, content="{}"))


if __name__ == "__main__":
    unittest.main()
