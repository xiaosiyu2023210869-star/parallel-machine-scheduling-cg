from __future__ import annotations

import ast
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceStructureTests(unittest.TestCase):
    def test_python_files_parse(self):
        for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_overridden_top_level_definitions(self):
        for path in sorted((ROOT / "src" / "pmcg").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            duplicates = [name for name, count in Counter(names).items() if count > 1]
            self.assertEqual([], duplicates, f"{path.name}: {duplicates}")

    def test_no_user_specific_absolute_paths(self):
        for path in sorted((ROOT / "src").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/mar/", text, path.name)


if __name__ == "__main__":
    unittest.main()
