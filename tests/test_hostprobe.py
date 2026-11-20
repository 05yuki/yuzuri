import importlib.util, os, unittest

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "hostprobe.py")
_s = importlib.util.spec_from_file_location("hostprobe", _p)
hostprobe = importlib.util.module_from_spec(_s); _s.loader.exec_module(hostprobe)

ZONEINFO = """Node 0, zone      DMA
  pages free     2816
        min      61
        low      76
        high     91
        spanned  4095
Node 0, zone   Normal
  pages free     1422852
        boost    0
        min      1301756
        low      1563579
        high     1825402
        nr_zone_active_anon 1
Node 1, zone   Normal
  pages free     100
        high     1000
"""


class Watermarks(unittest.TestCase):
    def test_sum_of_a_nodes_zones(self):
        w = hostprobe.watermarks(ZONEINFO, page=4096)
        self.assertEqual(w[0], (91 + 1825402) * 4096)
        self.assertEqual(w[1], 1000 * 4096)

    def test_empty(self):
        self.assertEqual(hostprobe.watermarks("", page=4096), {})

    def test_nodes_carry_them(self):
        for n in hostprobe.numa_nodes() if hasattr(hostprobe, "numa_nodes") else []:
            self.assertGreaterEqual(n["watermark_high"], 0)
            self.assertGreaterEqual(n["mem_kernel"], 0)


if __name__ == "__main__":
    unittest.main()
