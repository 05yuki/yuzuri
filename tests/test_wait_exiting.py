import importlib.util, io, os, tempfile, unittest
from contextlib import redirect_stdout

_here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")


def _load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(_here, f"{name}.py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


serve, hostprobe = _load("serve"), _load("hostprobe")
GONE = 2 ** 22 + 12345  # above the kernel's pid range: no such process


def host(exiting=(), in_use=()):
    return {"gpu_in_use": list(in_use), "exiting": [{"pid": p, "name": "sglang::schedul", "threads": 1} for p in exiting]}


class WaitForExiting(unittest.TestCase):
    def probes(self, seq):
        it = iter(seq)
        self.calls = 0

        def probe():
            self.calls += 1
            return next(it)
        return probe

    def test_nothing_exiting_does_not_probe_again(self):
        h = host()
        self.assertIs(serve.wait_for_exiting(self.probes([]), h), h)
        self.assertEqual(self.calls, 0)

    def test_waits_for_the_exiting_process_then_probes_again(self):
        done = host()
        out = io.StringIO()
        with redirect_stdout(out):
            got = serve.wait_for_exiting(self.probes([host([GONE]), done]), host([GONE]))
        self.assertIs(got, done)
        self.assertEqual(self.calls, 2)
        self.assertEqual(out.getvalue().count(f"sglang::schedul({GONE})"), 1)

    def test_a_server_on_the_cards_is_left_to_the_in_use_check(self):
        h = host([GONE], in_use=[{"pid": 1, "name": "sglang::scheduler"}])
        self.assertIs(serve.wait_for_exiting(self.probes([]), h), h)

    def test_plans_after_the_limit(self):
        h = host([GONE])
        out = io.StringIO()
        with redirect_stdout(out):
            got = serve.wait_for_exiting(self.probes([h] * 3), h, limit_s=0.0)
        self.assertIs(got, h)
        self.assertIn("still exiting", out.getvalue())


class Exiting(unittest.TestCase):
    def proc(self, tasks):
        """tasks: {pid: [(state, flags), ...]} as /proc/<pid>/task/<tid>/stat"""
        d = tempfile.mkdtemp()
        for pid, ts in tasks.items():
            os.makedirs(f"{d}/{pid}/task")
            open(f"{d}/{pid}/comm", "w").write(f"p{pid}\n")
            for k, (state, flags) in enumerate(ts):
                os.makedirs(f"{d}/{pid}/task/{pid + k}")
                open(f"{d}/{pid}/task/{pid + k}/stat", "w").write(
                    f"{pid + k} (p {pid}) {state} 1 1 1 0 -1 {flags} 0 0 0 0\n")
        os.makedirs(f"{d}/self")
        return d

    def test_threads_in_exit_are_listed_zombies_are_not(self):
        d = self.proc({100: [("S", 0x400000)],                 # running
                       200: [("Z", 0x404), ("D", 0x404)],      # leader a zombie, a thread still unmapping
                       300: [("Z", 0x40000c)],                 # an unreaped zombie
                       400: [("R", 0x4)]})                     # exiting
        self.assertEqual(hostprobe.exiting(d), [{"pid": 200, "name": "p200", "threads": 1},
                                                {"pid": 400, "name": "p400", "threads": 1}])


if __name__ == "__main__":
    unittest.main()
