import tempfile
import unittest
from pathlib import Path

from integration_model_smoke import read_first_gaussian_geometry


class GaussianGeometryParserTests(unittest.TestCase):
    def test_reads_first_link_cartesian_geometry(self):
        content = """%chk=a.chk
#p external='mlpint'

title

-1 2
C 0 0 0
H 0 0 1

--Link1--
%chk=a.chk
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.gjf"
            path.write_text(content, encoding="utf-8")
            charge, multiplicity, coordinates = read_first_gaussian_geometry(path)
        self.assertEqual((charge, multiplicity), (-1, 2))
        self.assertEqual(coordinates, ["C 0 0 0", "H 0 0 1"])


if __name__ == "__main__":
    unittest.main()

