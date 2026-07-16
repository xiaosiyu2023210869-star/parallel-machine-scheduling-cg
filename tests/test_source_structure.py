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

    def test_all_workflows_delegate_to_the_ml_fixed_k_cg(self):
        pmcg = ROOT / "src" / "pmcg"
        implementation_functions = {
            "build_initial_column_pool",
            "build_training_initial_column_pool",
            "build_rmp",
            "solve_pricing_multiple",
            "solve_integer_master",
            "build_final_k_sensitive_schedule_pool",
        }

        canonical_tree = ast.parse((pmcg / "ml_assisted.py").read_text(encoding="utf-8"))
        canonical_names = {
            node.name for node in canonical_tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(implementation_functions <= canonical_names)

        for filename in ("comparison.py", "small_comparison.py", "training.py"):
            tree = ast.parse((pmcg / filename).read_text(encoding="utf-8"))
            top_level = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
            self.assertFalse(implementation_functions & top_level, filename)

            entrypoint = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "column_generation_with_time_limit"
            )
            called_names = {
                node.func.id
                for node in ast.walk(entrypoint)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            self.assertIn("run_fixed_k_column_generation", called_names, filename)


if __name__ == "__main__":
    unittest.main()
