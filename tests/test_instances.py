from __future__ import annotations

import json
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmcg.parameters import effective_machine_count


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

    def test_machine_count_never_exceeds_job_count(self):
        self.assertEqual(3, effective_machine_count(3, 10))
        self.assertEqual(6, effective_machine_count(9, 6))
        with self.assertRaises(ValueError):
            effective_machine_count(0, 6)

    def test_machine_sensitivity_dataset_matches_paper(self):
        path = ROOT / "data" / "instances_m_sensitivity.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(18, len(records))
        self.assertTrue({860018, 860022, 860030} <= {record["id"] for record in records})
        self.assertEqual({600, 1800}, {record["time_limit"] for record in records})


if __name__ == "__main__":
    unittest.main()
