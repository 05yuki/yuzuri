"""Unit tests for MXFP4 E8M0 scale compression automation (Stage 1 + Stage 2)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import auto_profile  # noqa: E402


class TestScaleCompressionAutoProfile(unittest.TestCase):
    def test_mxfp4_format_env_has_scale_pack(self):
        self.assertEqual(auto_profile.FORMAT_ENV["MXFP4"].get("KT_GPU_STREAM_SCALE_PACK"), "1")
        self.assertNotIn("KT_GPU_STREAM_SCALE_PACK", auto_profile.FORMAT_ENV["NVFP4"])
        self.assertNotIn("KT_GPU_STREAM_SCALE_PACK", auto_profile.FORMAT_ENV["BF16"])
        self.assertNotIn("KT_GPU_STREAM_SCALE_PACK", auto_profile.FORMAT_ENV["LLAMAFILE"])

    def test_format_of_distinguishes_e8m0_mxfp4(self):
        method, why = auto_profile._format_of({"weight_packed": "U8", "weight_scale": "F8_E8M0"})
        self.assertEqual(method, "MXFP4")
        self.assertIn("F8_E8M0", why)

        method_nv, why_nv = auto_profile._format_of(
            {"weight": "U8", "weight_scale": "F8_E4M3", "weight_scale_2": "F32"}
        )
        self.assertEqual(method_nv, "NVFP4")
        self.assertNotIn("F8_E8M0", why_nv)

        method_bf, _ = auto_profile._format_of({"weight": "BF16"})
        self.assertEqual(method_bf, "BF16")

        method_fp8, _ = auto_profile._format_of({"weight": "F8_E4M3", "weight_scale_inv": "F32"})
        self.assertEqual(method_fp8, "FP8")


if __name__ == "__main__":
    unittest.main()
