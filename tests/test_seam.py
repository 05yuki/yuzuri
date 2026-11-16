"""The hook helper against a stand-in module: what happens when SGLang's function is the one a hook was written for,
when it gained a parameter, when it lost one, when the tree already carries the same change, and when a line a
rewrite looks for is gone. No SGLang, no GPU.

  python -m unittest discover -s tests
"""
import importlib.util
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from yuzuri.hooks import _seam  # noqa: E402

SOURCE = '''
def plain(a, b):
    return a + b

def grown(a, extra, b=2):
    return (a, extra, b)

def shrunk(a):
    return a

def carried(a, b):
    yuzuri_marker = 1
    return a + b

def body(x):
    y = x + 1
    return y * 2

def wrapped(x, table):
    y = table.get(
        x,
        0,
    )
    return y + 1

class Box:
    def method(self, a, b):
        return a * b
'''


def fake(test):
    """A module made from a file, as SGLang's are: inspect can read its source."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "seam_fake_%d.py" % len(sys.modules))
    with open(path, "w") as f:
        f.write(SOURCE)
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    test.addCleanup(sys.modules.pop, spec.name, None)
    return m


class Wrap(unittest.TestCase):
    def test_same_signature(self):
        m = fake(self)
        _seam.wrap(m, "plain", ["a", "b"], lambda orig: lambda a, b: orig(a, b) * 10)
        self.assertEqual(m.plain(1, 2), 30)

    def test_method(self):
        m = fake(self)
        _seam.wrap(m.Box, "method", ["self", "a", "b"], lambda orig: lambda self, a, b: orig(self, a, b) + 1)
        self.assertEqual(m.Box().method(3, 4), 13)

    def test_added_parameter_passes_through(self):
        m = fake(self)
        seen = []

        def make(orig):
            def hook(a, b=2):
                seen.append((a, b))
                return orig(a, b)
            return hook

        _seam.wrap(m, "grown", ["a", "b"], make)
        self.assertEqual(m.grown(1, "x", b=5), (1, "x", 5))
        self.assertEqual(m.grown(1, extra="y"), (1, "y", 2))
        self.assertEqual(seen, [(1, 5), (1, 2)])

    def test_lost_parameter_stops(self):
        m = fake(self)
        with self.assertRaises(_seam.SeamError) as e:
            _seam.wrap(m, "shrunk", ["a", "b"], lambda orig: orig)
        self.assertIn("'b'", str(e.exception))
        self.assertEqual(m.shrunk(4), 4)

    def test_missing_function_stops(self):
        m = fake(self)
        with self.assertRaises(_seam.SeamError):
            _seam.wrap(m, "nowhere", ["a"], lambda orig: orig)

    def test_tree_with_the_same_change_keeps_its_own(self):
        m = fake(self)
        before = m.carried
        _seam.wrap(m, "carried", ["a", "b"], lambda orig: lambda a, b: -1, adds=("yuzuri_marker",))
        self.assertIs(m.carried, before)
        self.assertEqual(m.carried(1, 2), 3)

    def test_tree_with_the_same_change_may_have_rewritten_what_the_hook_needs(self):
        m = fake(self)
        before = m.carried
        _seam.wrap(m, "carried", ["a", "b"], lambda orig: lambda a, b: -1, adds=("yuzuri_marker",),
                   needs=("a - b",))
        self.assertIs(m.carried, before)

    def test_needs_gone_stops(self):
        m = fake(self)
        with self.assertRaises(_seam.SeamError):
            _seam.wrap(m, "plain", ["a", "b"], lambda orig: orig, needs=("a - b",))

    def test_leading_names_only(self):
        m = fake(self)
        _seam.wrap(m, "grown", ["a", "..."], lambda orig: lambda a, *r, **k: orig(a + 1, *r, **k))
        self.assertEqual(m.grown(1, "x"), (2, "x", 2))


class Rewrite(unittest.TestCase):
    def test_edit_applied(self):
        m = fake(self)
        _seam.rewrite(m, "body", [("y = x + 1", "y = x + 100")])
        self.assertEqual(m.body(1), 202)

    def test_line_spaced_another_way_still_matches(self):
        m = fake(self)
        _seam.rewrite(m, "wrapped", [("    y = table.get( x, 0, )\n", "    y = table.get(x, 5)\n")])
        self.assertEqual(m.wrapped(1, {}), 6)

    def test_renamed_line_stops(self):
        m = fake(self)
        with self.assertRaises(_seam.SeamError):
            _seam.rewrite(m, "wrapped", [("    y = lookup.get( x, 0, )\n", "    y = 0\n")])
        self.assertEqual(m.wrapped(1, {}), 1)

    def test_line_gone_stops(self):
        m = fake(self)
        with self.assertRaises(_seam.SeamError):
            _seam.rewrite(m, "body", [("y = x + 7", "y = x")])
        self.assertEqual(m.body(1), 4)

    def test_tree_with_the_same_change_keeps_its_own(self):
        m = fake(self)
        _seam.rewrite(m, "carried", [("return a + b", "return 0")], adds=("yuzuri_marker",))
        self.assertEqual(m.carried(1, 2), 3)


class Define(unittest.TestCase):
    def test_new_name(self):
        m = fake(self)
        _seam.define(m, "def helper(x):\n    return plain(x, x)\n", "helper")
        self.assertEqual(m.helper(3), 6)

    def test_name_the_module_has_stays(self):
        m = fake(self)
        _seam.define(m, "def plain(a, b):\n    return 0\n", "plain")
        self.assertEqual(m.plain(1, 2), 3)


class Undo(unittest.TestCase):
    def test_failed_hook_is_taken_back(self):
        m = fake(self)
        before = m.plain
        mark = _seam.begin("test")
        _seam.wrap(m, "plain", ["a", "b"], lambda orig: lambda a, b: 0)
        self.assertEqual(m.plain(1, 2), 0)
        _seam.undo(mark)
        _seam.end(mark)
        self.assertIs(m.plain, before)


if __name__ == "__main__":
    unittest.main()
