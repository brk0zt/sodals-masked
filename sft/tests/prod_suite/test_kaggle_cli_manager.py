"""Unit tests for Kaggle CLI orchestrator (`kaggle_cli_manager.py`)."""

import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

SFT_DIR = Path(__file__).resolve().parents[2]
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

import kaggle_cli_manager as kcm


class _FakeApi:
    """In-memory fake matching `KaggleApi` methods exercised by `kaggle_cli_manager`."""

    def __init__(self, ds_exists: bool = False):
        self.ds_exists = ds_exists
        self.status_calls = 0
        self.pushed_folder = None
        self.pushed_acc = None
        self.ds_action = None

    def dataset_status(self, dataset_ref: str) -> str:
        if not self.ds_exists:
            raise RuntimeError("404 not found")
        return "ready"

    def dataset_create_new(self, folder: str, public: bool, quiet: bool, convert_to_csv: bool, dir_mode: str):
        assert dir_mode == "zip"
        assert convert_to_csv is False
        self.ds_exists = True
        self.ds_action = "new"
        return SimpleNamespace(error=None)

    def dataset_create_version(self, folder: str, version_notes: str, quiet: bool, convert_to_csv: bool, dir_mode: str):
        assert dir_mode == "zip"
        assert convert_to_csv is False
        self.ds_action = "version"
        return SimpleNamespace(error=None)

    def kernels_push(self, folder: str, acc: str = "", no_run: bool = False):
        self.pushed_folder = folder
        self.pushed_acc = acc
        return SimpleNamespace(error=None, url="https://www.kaggle.com/code/alice/svon-epoch-ii-and-iii-t4-x2", versionNumber=1)

    def kernels_status(self, kernel_ref: str):
        self.status_calls += 1
        return SimpleNamespace(status="COMPLETE", failure_message=None)

    def kernels_logs(self, kernel_ref: str) -> str:
        return "=== [EPOCH II.1] Done ===\n=== [EPOCH III] Done ==="

    def kernels_output(self, kernel_ref: str, path: str, force: bool = True, quiet: bool = False):
        zp = Path(path) / "svon_kaggle_artifacts.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("cloud_pipeline_summary.json", json.dumps({"ok": True}))
            zf.writestr("svon_cgw_checkpoints/svon_cgw_hybrid.qmem", b"QMEMmock")
        return ([str(zp)], "ok")


class TestKaggleCliManager(unittest.TestCase):
    """Validates staging, metadata generation, security, and pipeline orchestration."""

    def test_stage_bundle_and_metadata_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = kcm.stage_bundle(username="testuser", bundle_dir=Path(tmp) / "bundle")
            ds_dir = Path(info["dataset_dir"])
            kn_dir = Path(info["kernel_dir"])

            ds_meta = json.loads((ds_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
            kn_meta = json.loads((kn_dir / "kernel-metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(ds_meta["id"], "testuser/svon-epoch-ii-and-iii-code-bundle")
            self.assertEqual(kn_meta["id"], "testuser/svon-epoch-ii-and-iii-t4-x2")
            self.assertEqual(kn_meta["machine_shape"], "NvidiaTeslaT4")
            self.assertTrue(kn_meta["enable_gpu"])
            self.assertTrue(kn_meta["enable_internet"])
            self.assertEqual(kn_meta["kernel_type"], "notebook")
            self.assertEqual(kn_meta["dataset_sources"], ["testuser/svon-epoch-ii-and-iii-code-bundle"])
            self.assertTrue((ds_dir / "tests/prod_suite/test_svon_alphaedit_surgeon.py").is_file())
            self.assertTrue((ds_dir / "soft_lora_blender.py").is_file())
            self.assertTrue((kn_dir / kcm.NOTEBOOK_FILENAME).is_file())

            # Ensure mock adapter exists for test if not present in environment
            d1_dir = kcm.SFT_DIR / "qwen_lora_adapter_grpo_math_train_100"
            d2_dir = kcm.SFT_DIR / "qwen_lora_adapter_domain2_r8_pondering"
            created_mocks = []
            for d in (d1_dir, d2_dir):
                if not d.exists():
                    d.mkdir(parents=True, exist_ok=True)
                    created_mocks.append(d)
                for fn in ("adapter_config.json", "adapter_model.safetensors"):
                    fp = d / fn
                    if not fp.exists():
                        fp.write_text("{}", encoding="utf-8")
                        created_mocks.append(fp)
            try:
                info_prod = kcm.stage_bundle(username="testuser", bundle_dir=Path(tmp) / "bundle_prod", include_prod_adapters=True)
                ds_prod = Path(info_prod["dataset_dir"])
                self.assertTrue((ds_prod / "prod_adapters/d1/adapter_config.json").is_file())
                self.assertTrue((ds_prod / "prod_adapters/d2/adapter_config.json").is_file())
            finally:
                for item in reversed(created_mocks):
                    if item.is_file():
                        item.unlink(missing_ok=True)
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)

    def test_credentials_discovery_from_config_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "kaggle.json"
            cfg.write_text(json.dumps({"username": "burak_test", "key": "secret123"}), encoding="utf-8")
            creds = kcm.find_kaggle_credentials(config_dir=tmp)
            self.assertTrue(creds["found"])
            self.assertEqual(creds["username"], "burak_test")
            self.assertEqual(kcm.resolve_kaggle_username(config_dir=tmp), "burak_test")

    def test_access_token_discovery_from_config_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = Path(tmp) / "access_token"
            tok.write_text("KGAT_unit_test_token_123\n", encoding="utf-8")
            creds = kcm.find_kaggle_credentials(config_dir=tmp)
            self.assertTrue(creds["found"])
            self.assertEqual(creds["auth_type"], "access_token")
            self.assertEqual(creds["access_token"], "KGAT_unit_test_token_123")

    def test_safe_extract_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            zp = Path(tmp) / "evil.zip"
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("../escape.txt", "nope")
            with self.assertRaises(ValueError):
                kcm.safe_extract_zip(zp, Path(tmp) / "dest")

    def test_push_watch_download_end_to_end_mocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage_info = kcm.stage_bundle(username="alice", bundle_dir=Path(tmp) / "bundle")
            fake_api = _FakeApi(ds_exists=False)
            push_res = kcm.push_dataset_and_kernel(fake_api, stage_info, wait_dataset_sec=10)
            self.assertEqual(push_res["dataset_action"], "created")
            self.assertEqual(fake_api.pushed_acc, "NvidiaTeslaT4")

            out_dir = Path(tmp) / "dl"
            watch_res = kcm.watch_kernel(
                fake_api,
                push_res["kernel_ref"],
                poll_sec=1,
                max_wait_min=1,
                log_out_path=out_dir / "kernel.log",
            )
            self.assertEqual(watch_res["status"], "COMPLETE")
            self.assertTrue((out_dir / "kernel.log").is_file())

            dl_res = kcm.download_kernel_outputs(fake_api, push_res["kernel_ref"], out_dir)
            self.assertGreaterEqual(dl_res["extracted_count"], 2)
            self.assertTrue(dl_res["found_artifacts"]["cloud_pipeline_summary.json"])
            self.assertTrue(dl_res["found_artifacts"]["svon_cgw_hybrid.qmem"])


if __name__ == "__main__":
    unittest.main()
