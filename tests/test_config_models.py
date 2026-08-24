import tempfile
import unittest
from pathlib import Path

from qmint.config import default_config, load_config, save_config
from qmint.models import add_custom_model, resolve_model


class ConfigAndModelTests(unittest.TestCase):
    def test_config_round_trip_and_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = default_config()
            config["active_model"] = "uma-m"
            save_config(config, path)
            self.assertEqual(load_config(path)["active_model"], "uma-m")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_builtin_alias_resolves_backend_and_model_dir(self):
        config = default_config()
        config["model_dir"] = "/tmp/qmint-models"
        spec = resolve_model("small", config)
        self.assertEqual(spec.name, "uma-s")
        self.assertEqual(spec.backend, "fairchem")
        self.assertEqual(spec.path, Path("/tmp/qmint-models/uma-s-1p1.pt"))

    def test_custom_model_registration(self):
        config = default_config()
        add_custom_model(config, "research", "~/weights.model", "mace", "fine-tuned")
        spec = resolve_model("research", config)
        self.assertEqual(spec.backend, "mace")
        self.assertEqual(spec.description, "fine-tuned")

    def test_backend_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_model("uma-s", default_config(), "mace")


if __name__ == "__main__":
    unittest.main()
