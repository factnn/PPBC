"""Run from repro/CFD: python -m unittest discover -s tests."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.ppbc_drivaernetpp_transformer_easy import load_mean_std


class NormalizationTests(unittest.TestCase):
    def load(self, fields):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mean_std.json").write_text(json.dumps(fields))
            return load_mean_std(root)

    def fields(self):
        return {
            "pressure": {"mean": [-109.0], "std": [140.5]},
            "wss": {"mean": [0.65, -0.02, 0.08], "std": [0.82, 0.40, 0.77]},
        }

    def test_drivaernet_fields_only(self):
        original = self.fields()
        result = self.load(original)
        self.assertEqual(result["centroid"]["std"], [4.0])
        for field in original:
            self.assertEqual(result[field], original[field])

    def test_drivaerstar_coordinate_stats_preserved(self):
        original = self.fields()
        original["centroid"] = {"mean": [1.2, 0, 0.18], "std": [1.39, 0.57, 0.33]}
        self.assertEqual(self.load(original), original)

    def test_invalid_scale_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                fields = self.fields()
                fields["pressure"]["std"] = [value]
                with self.assertRaises(ValueError):
                    self.load(fields)

    def test_missing_physical_field_rejected(self):
        fields = self.fields()
        del fields["wss"]
        with self.assertRaises(KeyError):
            self.load(fields)


if __name__ == "__main__":
    unittest.main()
