"""The finder's handling of a tree that carries its own copy of a Yuzuri module: no GPU, no SGLang."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from yuzuri import _finder  # noqa: E402


class OwnModuleNumbers(unittest.TestCase):
    def _module(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_a_word_becomes_the_number_it_starts_from(self):
        path = self._module('import os\nT = int(os.environ.get("KT_GPU_STREAM_PREFILL", "0") or 0)\n')
        env = {"KT_GPU_STREAM_PREFILL": "auto", "KT_PREFILL_PLAN_ALIGN": "2048"}
        self.assertEqual(_finder._numbers_for("m", path, env), ["KT_GPU_STREAM_PREFILL"])
        self.assertEqual(env["KT_GPU_STREAM_PREFILL"], "2048")

    def test_a_number_is_left_alone(self):
        path = self._module('import os\nT = int(os.environ.get("KT_GPU_STREAM_PREFILL", "0"))\n')
        env = {"KT_GPU_STREAM_PREFILL": "4096"}
        self.assertEqual(_finder._numbers_for("m", path, env), [])
        self.assertEqual(env["KT_GPU_STREAM_PREFILL"], "4096")

    def test_a_word_without_a_number_stops_naming_the_switch(self):
        path = self._module('import os\nT = int(os.environ.get("KT_SOMETHING", "0"))\n')
        with self.assertRaisesRegex(RuntimeError, "KT_SOMETHING"):
            _finder._numbers_for("m", path, {"KT_SOMETHING": "auto"})

    def test_a_switch_the_module_does_not_read_as_a_number_is_left_alone(self):
        path = self._module('import os\nT = os.environ.get("KT_GPU_STREAM_PREFILL", "0")\n')
        env = {"KT_GPU_STREAM_PREFILL": "auto"}
        self.assertEqual(_finder._numbers_for("m", path, env), [])


if __name__ == "__main__":
    unittest.main()
