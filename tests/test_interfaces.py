import tempfile
import unittest
from pathlib import Path

import numpy as np

from qmint.interfaces.common import read_xyz, strip_suffix, write_engrad
from qmint.interfaces.orca_hessian import frequencies_and_modes


class InterfaceTests(unittest.TestCase):
    def test_orca_suffix_is_removed_exactly(self):
        self.assertEqual(strip_suffix("sample_EXT.extinp.tmp"), "sample_EXT")
        self.assertEqual(strip_suffix("attempt.tmp"), "attempt")

    def test_xyz_and_engrad(self):
        with tempfile.TemporaryDirectory() as directory:
            xyz = Path(directory) / "water.xyz"
            xyz.write_text("2\ncomment\nH 0 0 0\nH 0 0 1\n", encoding="utf-8")
            count, symbols, positions, lines = read_xyz(str(xyz))
            self.assertEqual((count, symbols), (2, ["H", "H"]))
            self.assertEqual(positions[1][2], 1.0)
            output = Path(directory) / "result.engrad"
            write_engrad(str(output), 2, -1.0, np.zeros((2, 3)), True)
            self.assertIn("-1.000000000000", output.read_text(encoding="utf-8"))

    def test_frequency_shapes(self):
        frequencies, modes = frequencies_and_modes(np.eye(6), [1.0, 1.0])
        self.assertEqual(frequencies.shape, (6,))
        self.assertEqual(modes.shape, (6, 6))


if __name__ == "__main__":
    unittest.main()

