import unittest

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree

from qmint.calculator import calculate


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.results = {
            "energy": 0.5 * float(np.sum(positions**2)),
            "forces": -positions,
        }

    def get_hessian(self, atoms):
        return np.eye(3 * len(atoms))


class CalculatorTests(unittest.TestCase):
    def test_analytic_task_shape(self):
        result = calculate(
            {
                "coord": ["H 0 0 0", "H 0 0 1"],
                "charge": 0,
                "mult": 1,
                "num_threads": 1,
                "hessian": True,
            },
            HarmonicCalculator(),
            "analytic",
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["gradient"].shape, (2, 3))
        self.assertEqual(result["hessian"].shape, (6, 6))
        self.assertAlmostEqual(result["hessian"][0, 0], 1.0 / (Hartree / (Bohr * Bohr)))


if __name__ == "__main__":
    unittest.main()
