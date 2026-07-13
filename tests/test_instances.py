from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class InstanceDataTests(unittest.TestCase):
    def test_all_instance_files_are_valid(self):
        for path in sorted((ROOT / "data").glob("instances_*.json")):
            records = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(records, path.name)
            seen = set()
            for record in records:
                self.assertNotIn(record["id"], seen, path.name)
                seen.add(record["id"])
                self.assertEqual(record["n"], len(record["p"]), path.name)
                self.assertEqual(len(record["p"]), len(record["w"]), path.name)
                self.assertTrue(all(value > 0 for value in record["p"]), path.name)
                self.assertTrue(all(value > 0 for value in record["w"]), path.name)


if __name__ == "__main__":
    unittest.main()
