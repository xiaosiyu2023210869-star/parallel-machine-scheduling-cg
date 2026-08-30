from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SensitivityPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_script("run_theta_c_sensitivity.py")
        cls.extension = load_script("run_high_rho_extension.py")

    def test_primary_and_interaction_plan_has_111_cases(self):
        args = type(
            "Args",
            (),
            {"value_min": 1, "value_max": 36, "random_seed": 20260819},
        )()
        instances = self.main.load_instances(args)
        cases = self.main.build_cases(instances, 3, self.main.INTERACTION_N)
        self.assertEqual(25, len(instances))
        self.assertEqual(111, len(cases))
        self.assertEqual(75, sum(case.experiment == "theta_sensitivity" for case in cases))
        self.assertEqual(36, sum(case.experiment == "theta_c_interaction" for case in cases))

    def test_high_rho_plan_matches_final_paper(self):
        instances = self.extension.load_instances(20260819, 1, 36, False)
        cases = self.extension.build_case_plan(instances, 3)
        self.assertEqual((6, 16, 40, 110, 150, 200), self.extension.REPRESENTATIVE_N)
        self.assertEqual(12, len(cases))
        self.assertEqual({3.0, 4.0}, {case.theta_ratio for case in cases})
        self.assertEqual({4}, {case.c_value for case in cases if case.n <= 36})
        self.assertEqual({2}, {case.c_value for case in cases if case.n > 36})


if __name__ == "__main__":
    unittest.main()
