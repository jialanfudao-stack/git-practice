import hashlib
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import qwen_drive_cache as cache


PAYLOAD = b"small pinned qwen model fixture"
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class DriveCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mydrive = self.root / "MyDrive"
        self.mydrive.mkdir()
        self.local = self.root / "runtime" / "qwen" / "model.gguf"
        self.downloads = 0

    def tearDown(self):
        self.temp.cleanup()

    def download(self, *, repo_id, filename, revision, local_dir):
        self.downloads += 1
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PAYLOAD)
        return path

    def ensure(self, local=None, download=None, **overrides):
        args = {
            "mydrive_root": self.mydrive,
            "local_path": local or self.local,
            "repo_id": "owner/model",
            "revision": "a" * 40,
            "filename": "model.gguf",
            "expected_size": len(PAYLOAD),
            "expected_sha256": SHA256,
            "download": download or self.download,
        }
        args.update(overrides)
        return cache.ensure_model(**args)

    def drive_file(self):
        return (
            self.mydrive
            / "QwenModels"
            / "huggingface"
            / "owner"
            / "model"
            / ("a" * 40)
            / SHA256
            / "model.gguf"
        )

    def test_first_download_promotes_verified_drive_copy_and_reuses_cache(self):
        self.ensure()
        self.assertEqual(self.downloads, 1)
        self.assertEqual(self.local.read_bytes(), PAYLOAD)
        self.assertEqual(self.drive_file().read_bytes(), PAYLOAD)
        self.assertTrue(cache._stamp_valid(
            self.drive_file(), "owner/model", "a" * 40, SHA256, len(PAYLOAD)
        ))

        self.ensure()
        self.assertEqual(self.downloads, 1)

    def test_new_runtime_restores_from_drive_without_hugging_face_download(self):
        self.ensure()
        self.local.unlink()
        cache._stamp_path(self.local).unlink()
        restored = self.ensure()
        self.assertEqual(restored.read_bytes(), PAYLOAD)
        self.assertEqual(self.downloads, 1)

    def test_existing_good_local_model_is_migrated_without_download(self):
        self.local.parent.mkdir(parents=True)
        self.local.write_bytes(PAYLOAD)
        self.ensure()
        self.assertEqual(self.downloads, 0)
        self.assertEqual(self.drive_file().read_bytes(), PAYLOAD)
        self.assertEqual(self.local.read_bytes(), PAYLOAD)

    def test_corrupt_drive_copy_is_replaced_from_good_local_copy(self):
        self.ensure()
        self.drive_file().write_bytes(b"corrupt")
        self.ensure()
        self.assertEqual(self.downloads, 1)
        self.assertEqual(self.drive_file().read_bytes(), PAYLOAD)

    def test_corrupt_local_copy_is_restored_from_drive(self):
        self.ensure()
        self.local.write_bytes(b"corrupt")
        self.ensure()
        self.assertEqual(self.downloads, 1)
        self.assertEqual(self.local.read_bytes(), PAYLOAD)

    def test_interrupted_bad_download_never_gets_a_verified_stamp(self):
        def bad_download(*, repo_id, filename, revision, local_dir):
            path = Path(local_dir) / filename
            path.write_bytes(b"partial")
            return path

        with self.assertRaises(cache.ModelIntegrityError):
            self.ensure(download=bad_download)
        self.assertFalse(self.local.exists())
        self.assertFalse(self.drive_file().exists())
        self.assertFalse(list(self.local.parent.glob("*.partial")))

    def test_missing_mydrive_stops_instead_of_using_ephemeral_cache(self):
        with self.assertRaisesRegex(RuntimeError, "MyDrive mount is missing"):
            self.ensure(mydrive_root=self.root / "not-mounted")
        self.assertEqual(self.downloads, 0)

    def test_drive_mount_requires_mounted_root_and_mydrive_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "drive"
            root.mkdir()
            (root / "MyDrive").mkdir()
            with patch.object(cache, "_mountpoint", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "not mounted"):
                    cache.mounted_mydrive(root)

    def test_drive_io_error_does_not_fall_back_to_hugging_face_or_delete_local_good(self):
        self.local.parent.mkdir(parents=True)
        self.local.write_bytes(PAYLOAD)

        def fail_copy(*_args, **_kwargs):
            raise OSError("simulated Drive full")

        with patch.object(cache, "_copy_verified", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "Drive full"):
                self.ensure()
        self.assertEqual(self.local.read_bytes(), PAYLOAD)
        self.assertEqual(self.downloads, 0)

    def test_image_cell_preserves_hugging_face_repo_relative_filename(self):
        notebook = json.loads(Path(__file__).with_name("Untitled0.ipynb").read_text(encoding="utf-8"))
        namespace = {
            "Path": Path,
            "QWEN_DRIVE_CACHE": cache,
            "QWEN_MYDRIVE_ROOT": self.mydrive,
            "GPU_INFERENCE_LOCK": None,
            "download_calls": [],
        }

        def mock_download(repo_id, filename, revision, local_dir):
            namespace["download_calls"].append((repo_id, filename, revision))
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(PAYLOAD)
            return path

        namespace["_hf_pinned_download"] = mock_download
        for cell_index, function_name in ((6, "_assert_model_io_allowed"), (6, "_ensure_pinned_model"), (17, "_ensure_image_model")):
            source = "".join(notebook["cells"][cell_index]["source"])
            tree = ast.parse(source)
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
            module = ast.Module(body=[function], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), f"<cell {cell_index}:{function_name}>", "exec"), namespace)

        local_path = self.root / "runtime" / "ComfyUI" / "models" / "text_encoders" / "clip.safetensors"
        namespace["_ensure_image_model"](
            "Comfy-Org/Qwen-Image-2.1",
            "text_encoders/clip.safetensors",
            "b" * 40,
            local_path,
            len(PAYLOAD),
            SHA256,
        )
        self.assertEqual(
            namespace["download_calls"],
            [("Comfy-Org/Qwen-Image-2.1", "text_encoders/clip.safetensors", "b" * 40)],
        )


if __name__ == "__main__":
    unittest.main()
