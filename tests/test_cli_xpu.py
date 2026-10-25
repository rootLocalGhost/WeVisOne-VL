"""Unit tests for wevisone_vl_xpu.cli using standard unittest library."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from wevisone_vl_xpu import cli


class _FakeEngine:
    error: Exception | None = None
    last_kwargs: dict = {}

    def __init__(self, *args, **kwargs) -> None:
        _FakeEngine.last_kwargs = kwargs

    def predict(self, image, **kwargs):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(response="ok", prompt="test", boxes=[], masks=[], latency_s=0.1)


class TestCliXPU(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.image_path = self.tmp_path / "input.png"
        Image.new("RGB", (8, 8), "white").save(self.image_path)

        _FakeEngine.error = None
        _FakeEngine.last_kwargs = {}

        self.patcher_engine = patch.object(cli, "WeVisOneVL", _FakeEngine)
        self.patcher_save = patch.object(
            cli,
            "save_result",
            lambda output_dir, image, result: {"metadata": Path(output_dir) / "metadata.json"},
        )
        self.patcher_engine.start()
        self.patcher_save.start()

    def tearDown(self):
        self.patcher_engine.stop()
        self.patcher_save.stop()
        self.tmp_dir.cleanup()

    def _run(self, *extra: str) -> int:
        args = ["wevisone-vl-xpu", "--image", str(self.image_path), "--output-dir", str(self.tmp_path / "out"), *extra]
        with patch.object(sys, "argv", args):
            return cli.main()

    def test_success_returns_zero(self):
        code = self._run()
        self.assertEqual(code, 0)

    def test_sequential_loading_flag(self):
        code = self._run("--sequential-loading", "--device", "xpu:0")
        self.assertEqual(code, 0)
        self.assertTrue(_FakeEngine.last_kwargs.get("sequential_loading"))
        self.assertEqual(_FakeEngine.last_kwargs.get("device"), "xpu:0")

    def test_missing_image_is_a_usage_error(self):
        args = ["wevisone-vl-xpu", "--image", str(self.tmp_path / "missing.png"), "--output-dir", str(self.tmp_path / "out")]
        with patch.object(sys, "argv", args):
            code = cli.main()
        self.assertEqual(code, cli.EXIT_USAGE)

    def test_invalid_numeric_arguments_are_rejected(self):
        for flag in [("--max-new-tokens", "0"), ("--decoder-batch-size", "-1"), ("--temperature", "-0.1")]:
            with self.subTest(flag=flag):
                code = self._run(*flag)
                self.assertEqual(code, cli.EXIT_USAGE)

    def test_corrupt_image_is_reported_without_a_traceback(self):
        bad = self.tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        args = ["wevisone-vl-xpu", "--image", str(bad), "--output-dir", str(self.tmp_path / "out")]
        with patch.object(sys, "argv", args):
            code = cli.main()
        self.assertEqual(code, cli.EXIT_FAILURE)

    def test_model_init_failure_is_logged(self):
        def broken(*args, **kwargs):
            raise OSError("weights not found")

        with patch.object(cli, "WeVisOneVL", broken):
            code = self._run()
        self.assertEqual(code, cli.EXIT_FAILURE)

    def test_inference_failure_is_logged(self):
        _FakeEngine.error = RuntimeError("XPU out of memory")
        code = self._run()
        self.assertEqual(code, cli.EXIT_FAILURE)

    def test_save_failure_still_prints_the_response(self):
        def broken(*args, **kwargs):
            raise OSError("disk full")

        with patch.object(cli, "save_result", broken):
            code = self._run()
        self.assertEqual(code, cli.EXIT_FAILURE)

    def test_keyboard_interrupt_during_inference(self):
        _FakeEngine.error = KeyboardInterrupt()
        code = self._run()
        self.assertEqual(code, cli.EXIT_INTERRUPTED)


if __name__ == "__main__":
    unittest.main()
