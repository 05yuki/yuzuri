import importlib.util, io, os, unittest
from contextlib import redirect_stdout

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "serve.py")
_s = importlib.util.spec_from_file_location("serve", _p)
serve = importlib.util.module_from_spec(_s); _s.loader.exec_module(serve)

GIB = 1 << 30
GONE = 2 ** 22 + 12345  # above the kernel's pid range: no such process


def host(unlisted, pids=(), in_use=()):
    return {"gpu_in_use": list(in_use),
            "gpu_leaving": [{"gpu_uuid": "GPU-a", "unlisted": unlisted,
                             "pids": [{"pid": p, "name": "python"} for p in pids]}]}


class WaitForLeaving(unittest.TestCase):
    def probes(self, seq):
        it = iter(seq)
        self.calls = 0

        def probe():
            self.calls += 1
            return next(it)
        return probe

    def test_nothing_held_probes_once(self):
        h = host(80 << 20)
        with redirect_stdout(io.StringIO()):
            self.assertIs(serve.wait_for_leaving(self.probes([h])), h)
        self.assertEqual(self.calls, 1)

    def test_waits_for_the_holder_then_probes_again(self):
        done = host(30 << 20)
        out = io.StringIO()
        with redirect_stdout(out):
            got = serve.wait_for_leaving(self.probes([host(14 * GIB, [GONE]), done]))
        self.assertIs(got, done)
        self.assertEqual(self.calls, 2)
        self.assertIn("14.0 GiB", out.getvalue())
        self.assertIn(f"python({GONE})", out.getvalue())

    def test_a_listed_server_is_left_to_the_in_use_check(self):
        h = host(14 * GIB, [GONE], in_use=[{"pid": 1, "name": "sglang::scheduler"}])
        with redirect_stdout(io.StringIO()):
            self.assertIs(serve.wait_for_leaving(self.probes([h])), h)

    def test_stops_with_the_numbers_after_the_limit(self):
        h = host(14 * GIB, [GONE])
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as e:
            serve.wait_for_leaving(self.probes([h] * 10), limit_s=0.0)
        self.assertIn("14.0 GiB", str(e.exception))


if __name__ == "__main__":
    unittest.main()
