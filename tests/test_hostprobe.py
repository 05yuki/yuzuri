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

class Visible(unittest.TestCase):
    G = [{"index": 0, "uuid": "GPU-aaaa-1", "numa_node": 0}, {"index": 1, "uuid": "GPU-bbbb-2", "numa_node": 1}]

    def test_unset_is_every_card(self):
        self.assertEqual(hostprobe.visible(self.G, None), self.G)
        self.assertEqual(hostprobe.visible(self.G, ""), self.G)

    def test_an_index_picks_that_card(self):
        self.assertEqual([g["numa_node"] for g in hostprobe.visible(self.G, "1")], [1])

    def test_order_is_the_callers(self):
        self.assertEqual([g["index"] for g in hostprobe.visible(self.G, "1,0")], [1, 0])

    def test_a_uuid_prefix(self):
        self.assertEqual([g["index"] for g in hostprobe.visible(self.G, "GPU-bbbb")], [1])

    def test_a_card_that_is_not_there_stops(self):
        with self.assertRaises(SystemExit):
            hostprobe.visible(self.G, "2")


if __name__ == "__main__":
    unittest.main()
