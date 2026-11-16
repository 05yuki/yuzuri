"""What the tables say against what is in the tree: every hook listed exists, every feature's requirement is a
feature, every module of the manifest is a file, every profile is a launch the launcher can fill, the server
listens on this machine unless told otherwise, the .pth stays silent once the checkout is gone, and every Python
file compiles. No SGLang, no GPU.
"""
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "python", "yuzuri")
sys.path.insert(0, os.path.join(ROOT, "python"))


def text(*parts):
    with open(os.path.join(*parts), errors="replace") as f:
        return f.read()


def load(name):
    with open(os.path.join(PKG, name)) as f:
        return json.load(f)


class Tables(unittest.TestCase):
    def test_hooks_listed_exist(self):
        import yuzuri._features as F
        import yuzuri.hooks as H

        files = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(PKG, "hooks", "*.py"))} - {"__init__", "_seam"}
        imported = {n for n in vars(H) if n in files}
        self.assertEqual(files - imported, set(), "hook files hooks/__init__.py does not import")
        self.assertEqual(set(F.TABLE) - files, set(), "the feature report names hooks that are not files")

    def test_feature_requirements(self):
        feats = load("features.json")["features"]
        names = [f["name"] for f in feats]
        self.assertEqual(len(names), len(set(names)))
        for f in feats:
            for r in f.get("requires", []):
                self.assertIn(r, names, f"{f['name']} requires {r}")
            self.assertTrue(f.get("for"), f["name"])

    def test_manifest_modules_are_files(self):
        m = load("manifest.json")
        for e in m["modules"]:
            self.assertTrue(os.path.isfile(os.path.join(PKG, "tree", e["path"])), e["path"])
            self.assertEqual(e["module"], "sglang." + e["path"][:-3].replace("/", "."))

    def test_profiles(self):
        paths = glob.glob(os.path.join(ROOT, "profiles", "*.json"))
        self.assertTrue(paths)
        for p in paths:
            with open(p) as f:
                prof = json.load(f)
            cmd = prof["command"]
            self.assertEqual(cmd[cmd.index("--host") + 1], "{host}", p)
            self.assertNotIn("0.0.0.0", json.dumps(prof), p)

    def test_listens_on_this_machine_by_default(self):
        src = text(ROOT, "tools", "serve.py")
        self.assertRegex(src, r'add_argument\("--host", default="127\.0\.0\.1"')
        self.assertNotIn('"0.0.0.0"', text(ROOT, "tools", "auto_profile.py"))

    def test_everything_compiles(self):
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
        self.assertTrue(out)
        for rel in out:
            compile(text(ROOT, rel), rel, "exec")

    def test_licenses_carried(self):
        mit = text(ROOT, "LICENSES", "MIT-llama.cpp.txt")
        self.assertIn("Permission is hereby granted", mit)
        beside = os.path.join(PKG, "tree", "srt", "layers", "moe", "kt_gguf_moe_mmq", "llama_mmq", "ggml", "LICENSE")
        self.assertEqual(text(beside), mit)

    def test_docs_name_files_that_exist(self):
        """A tools/x.py, patches/x.py or hooks/x.py written in a README, a docstring or a comment is in the tree."""
        out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True).stdout.split()
        have = set(out) | {os.path.join("python", "yuzuri", "hooks", os.path.basename(p)) for p in out}
        missing = []
        for rel in out:
            if rel.startswith("tests/") or not (rel.endswith((".py", ".md", ".sh", ".json")) or rel.endswith("NOTICE")):
                continue
            for m in re.finditer(r"(?<![\w/.<>-])(tools|patches|hooks)/([\w.-]+\.(?:py|sh))", text(ROOT, rel)):
                path = os.path.join("python", "yuzuri", "hooks", m.group(2)) if m.group(1) == "hooks" else m.group(0)
                if path not in have:
                    missing.append(f"{rel}: {m.group(0)}")
        allowed = {"patches/kt-kernel-nvfp4-scale-codes.py: patches/kt-nvfp4-scale-codes.py"}  # a string it replaces
        self.assertEqual(sorted(set(missing) - allowed), [])

    def test_environment_list_matches_the_code(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "list-env.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)


class Pth(unittest.TestCase):
    def test_silent_once_the_checkout_is_gone(self):
        d = tempfile.mkdtemp()
        venv = os.path.join(d, "v")
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "install-pth.py"), venv], check=True, capture_output=True)
        pth = glob.glob(os.path.join(venv, "lib", "python3*", "site-packages", "yuzuri.pth"))[0]
        py = os.path.join(venv, "bin", "python")
        r = subprocess.run([py, "-c", "print('up')"], capture_output=True, text=True)
        self.assertEqual((r.stdout.strip(), r.stderr), ("up", ""))
        with open(pth) as f:
            text = f.read()
        with open(pth, "w") as f:
            f.write(text.replace(os.path.join(ROOT, "python"), os.path.join(d, "gone")))
        r = subprocess.run([py, "-c", "print('up')"], capture_output=True, text=True)
        self.assertEqual((r.stdout.strip(), r.stderr), ("up", ""))
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "install-pth.py"), venv, "--remove"], check=True, capture_output=True)
        self.assertFalse(os.path.exists(pth))


if __name__ == "__main__":
    unittest.main()
