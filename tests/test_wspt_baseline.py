from __future__ import annotations

import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmcg.wspt_baseline import (
    block_reordered_sequence,
    compute_sequence_cost,
    run_improved_wspt,
    validate_schedule,
)


class WsptBaselineTests(unittest.TestCase):
    def test_sequence_cost_includes_periodic_tool_change(self):
        cost = compute_sequence_cost([0, 1, 2], [2, 3, 5], [1, 2, 4], c=2, theta=7)
        self.assertEqual(2 + 2 * 5 + 4 * 17, cost)

    def test_block_reordering_uses_batch_wspt_index(self):
        sequence = [0, 1, 2, 3]
        reordered = block_reordered_sequence(
            sequence,
            p=[8, 8, 1, 1],
            w=[1, 1, 8, 8],
            c=2,
            theta=4,
        )
        self.assertEqual([2, 3, 0, 1], reordered)

    def test_heuristic_returns_a_complete_feasible_schedule(self):
        result = run_improved_wspt(
            p=[5, 2, 7, 3, 4],
            w=[2, 8, 1, 5, 3],
            m=3,
            c=2,
            theta=4,
        )
        validate_schedule(result.schedule, 5)
        self.assertGreater(result.objective, 0)
        self.assertGreater(result.wspt_lrf_objective, 0)


if __name__ == "__main__":
    unittest.main()
