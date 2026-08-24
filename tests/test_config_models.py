import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qmint.config import default_config, load_config, save_config
from qmint.models import ModelSpec, add_custom_model, download_model, resolve_model


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
        add_custom_model(
            config, "research", "~/weights.model", "mace", "fine-tuned", "omol"
        )
        spec = resolve_model("research", config)
        self.assertEqual(spec.backend, "mace")
        self.assertEqual(spec.description, "fine-tuned")
        self.assertEqual(spec.head, "omol")

    def test_multi_head_builtin_selects_omol(self):
        spec = resolve_model("mace-mh-1", default_config())
        self.assertEqual(spec.head, "omol")

    def test_backend_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_model("uma-s", default_config(), "mace")

    def test_public_models_include_verified_downloads(self):
        for name in ("mace-omol", "mace-polar-m", "mace-polar-l", "orbmol-v2"):
            spec = resolve_model(name, default_config())
            self.assertTrue(spec.download_url.startswith("https://"))
            self.assertEqual(len(spec.sha256), 64)
        self.assertIsNone(resolve_model("uma-s", default_config()).download_url)

    def test_model_download_is_atomic_and_verified(self):
        content = b"QMint model fixture"
        checksum = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            spec = ModelSpec(
                "fixture",
                "mace",
                path,
                download_url="https://example.test/model.bin",
                sha256=checksum,
            )
            with patch("urllib.request.urlopen", return_value=io.BytesIO(content)):
                self.assertEqual(download_model(spec), path)
            self.assertEqual(path.read_bytes(), content)

    def test_uma_requires_manual_download(self):
        spec = resolve_model("uma-s", default_config())
        with self.assertRaisesRegex(ValueError, "no automatic download URL"):
            download_model(spec)


if __name__ == "__main__":
    unittest.main()
