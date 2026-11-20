"""The prefill plan's cost: read from the forwards a server measured, by chunk size and by sequence. The forwards
are the ones V4.1 measured (11-18, 11-19). No torch, no SGLang, no GPU.
"""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "python", "yuzuri", "tree", "srt", "layers", "moe", "kt_prefill_plan.py")
M = 1 << 20


def plan(runs=()):
    for k in ("KT_PREFILL_PLAN_COST", "KT_PREFILL_PLAN_ROW_KB"):
        os.environ.pop(k, None)
    spec = importlib.util.spec_from_file_location("kt_prefill_plan_under_test", PLAN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for run in runs:
        for r, s, g, _ in run:
            m._fold(r, s, g * M, 0)
    m._refresh()
    return m


# (rows, sequence, MiB grown, MiB allocated) of each forward
# a 131K prompt on two cards: flat from its third chunk
R131 = [(22016, 22016, 7282, 4807), (22016, 44032, 8601, 5744), (22016, 66048, 8799, 5949),
        (22016, 88064, 8010, 5927), (22016, 110080, 8784, 5918), (20992, 131072, 8592, 5831)]
# a launch's first prompts on one card: two probes, then a 65K prompt still growing at its end
PROBE = [(1024, 1024, 632, 569), (4096, 4096, 1553, 1281), (8192, 8192, 2914, 2353), (8192, 16384, 2921, 2360)]
R65 = [(16384, 16384, 5601, 4390), (16384, 32768, 5676, 4395), (16384, 49152, 7220, 4733), (16384, 65536, 7739, 5250)]

# a 524K prompt on one card of one socket, 5,120-row chunks: (sequence, MiB grown, MiB allocated) of each forward.
# The allocation grew from 36K to 205K tokens and stayed, in steps as the rows scored at once changed (11-18)
R524 = [(5120, 2250, 1881), (10240, 2194, 1882), (15360, 2195, 1883), (20480, 2194, 1883), (25600, 2188, 1883),
        (30720, 2197, 1884), (35840, 2541, 1884), (40960, 2602, 1910), (46080, 2646, 1961), (51200, 2703, 2013),
        (56320, 2748, 2062), (61440, 2800, 2114), (66560, 2847, 2164), (71680, 2901, 2214), (76800, 2949, 2265),
        (81920, 3002, 2315), (87040, 3046, 2367), (92160, 3103, 2417), (97280, 3148, 2468), (102400, 3206, 2518),
        (107520, 3274, 2570), (112640, 3307, 2619), (117760, 3383, 2671), (122880, 3406, 2721), (128000, 3458, 2772),
        (133120, 3527, 2822), (138240, 3605, 2873), (143360, 3630, 2923), (148480, 3706, 2975), (153600, 3727, 3026),
        (158720, 3803, 3076), (163840, 3830, 3126), (168960, 3924, 3177), (174080, 3929, 3228), (179200, 4028, 3280),
        (184320, 4035, 3329), (189440, 4100, 3380), (194560, 4135, 3431), (199680, 4211, 3481), (204800, 4254, 3533),
        (209920, 4291, 3550), (215040, 4209, 3487), (220160, 4264, 3533), (225280, 4300, 3582), (230400, 4244, 3504),
        (235520, 4271, 3548), (240640, 4203, 3465), (245760, 4228, 3506), (250880, 4275, 3549), (256000, 4177, 3453),
        (261120, 4198, 3491), (266240, 4242, 3531), (271360, 4291, 3569), (276480, 4173, 3463), (281600, 4228, 3497),
        (286720, 4246, 3533), (291840, 4300, 3570), (296960, 4175, 3448), (302080, 4208, 3482), (307200, 4242, 3516),
        (312320, 4276, 3550), (317440, 4310, 3581), (322560, 4161, 3442), (327680, 4200, 3474), (332800, 4223, 3505),
        (337920, 4261, 3535), (343040, 4283, 3565), (348160, 4325, 3597), (353280, 4159, 3438), (358400, 4195, 3466),
        (363520, 4214, 3494), (368640, 4254, 3521), (373760, 4282, 3550), (378880, 4294, 3579), (384000, 4151, 3401),
        (389120, 4144, 3426), (394240, 4203, 3451), (399360, 4192, 3477), (404480, 4249, 3502), (409600, 4260, 3528),
        (414720, 4269, 3553), (419840, 4309, 3577), (424960, 4321, 3604), (430080, 4131, 3399), (435200, 4136, 3421),
        (440320, 4174, 3445), (445440, 4192, 3467), (450560, 4239, 3490), (455680, 4244, 3512), (460800, 4268, 3534),
        (465920, 4290, 3557), (471040, 4312, 3578), (476160, 4338, 3601), (481280, 4098, 3367), (486400, 4101, 3387),
        (491520, 4138, 3407), (496640, 4160, 3426), (501760, 4181, 3446), (506880, 4177, 3466), (512000, 4222, 3487),
        (517120, 4219, 3507), (522240, 4257, 3526)]
R524 = [(5120, q, g, a) for q, g, a in R524]
R16 = [(16384, 16384, 5188, 4488), (16384, 32768, 5193, 4493), (16384, 49152, 5531, 4831),
       (16384, 65536, 6049, 5349)]


class Measured(unittest.TestCase):
    def test_a_measured_size_costs_the_most_it_grew_up_to_a_sequence(self):
        m = plan([R131[:5]])
        self.assertEqual(m._own(22016, 22016), 7282 * M)
        self.assertEqual(m._own(22016, 44032), 8601 * M)
        self.assertEqual(m._own(22016, 110080), 8799 * M)   # the fourth chunk grew less than the third

    def test_a_flat_run_costs_the_same_past_its_sequence(self):
        m = plan([R131])
        self.assertEqual(m._own(22016, 524288), 8799 * M)
        self.assertEqual(m._own(22016, 1000000), 8799 * M)

    def test_a_run_still_growing_goes_on_growing(self):
        m = plan([PROBE, R65])
        # 7,220 to 7,739 MiB over its last third, 16,384 tokens
        self.assertAlmostEqual(m._own(16384, 131072) / M, 7739 + 519 * 4, delta=1)

    def test_a_long_run_that_went_flat_stays(self):
        m = plan([R524])
        self.assertGreaterEqual(m._own(5120, 204800), 4254 * M)
        self.assertLess(m._own(5120, 204800), 4300 * M)
        self.assertEqual(m._own(5120, 522240), 4338 * M)
        self.assertEqual(m._own(5120, 1000000), 4338 * M)
        self.assertLessEqual(len(m.TABLE[5120]), m.POINTS)
        self.assertEqual(m.TABLE[5120][0], (5120, 2250 * M))
        self.assertEqual(max(g for _, g in m.TABLE[5120]), 4338 * M)    # the costliest forward stays
        self.assertEqual(m.TABLE[5120][-1][0], 522240)

    def test_another_size_rises_as_the_longest_measured_one_did(self):
        m = plan([R524, R65])
        self.assertEqual(m.REF[0], 5120)
        # the 5,120-row run rose 4,338 - 2,800 MiB past 65,536, and a 16,384-row forward has 3.2 times its rows
        self.assertAlmostEqual(m._own(16384, 524288) / M, 7739 + (4338 - 2800) * 16384 / 5120, delta=64)

    def test_nothing_measured_costs_nothing(self):
        m = plan()
        self.assertEqual(m._own(4096, 4096), 0)
        self.assertEqual(m.online_cap(32768), 32768)


class Sizes(unittest.TestCase):
    def test_between_two_sizes_the_line(self):
        m = plan([PROBE])
        self.assertEqual(m._own(2560, 1024), (632 + 1553) / 2 * M)

    def test_under_the_smallest_its_cost_a_row(self):
        m = plan([PROBE])
        self.assertEqual(m._own(512, 512), 316 * M)
        self.assertEqual(m.ROW, 632 * M / 1024)

    def test_past_the_largest_no_less_than_its_cost_a_row(self):
        m = plan([PROBE, R65])
        self.assertAlmostEqual(m._own(20480, 65536) / M, 7739 * 20480 / 16384, delta=1)

    def test_past_the_largest_the_last_slope_where_it_is_steeper(self):
        m = plan([[(4096, 4096, 1000, 0), (8192, 8192, 4000, 0)]])
        self.assertEqual(m._own(12288, 8192), 7000 * M)

    def test_a_larger_chunk_costs_no_less_than_a_smaller_one(self):
        m = plan([PROBE, R65, R131])
        last = 0
        for c in range(1024, 30000, 1024):
            v = m._own(c, 131072)
            self.assertGreaterEqual(v, last)
            last = v

    def test_the_cap_is_twice_the_largest_measured(self):
        m = plan([PROBE])
        self.assertEqual(m.online_cap(65536), 16384)

    def test_many_sizes_fold_into_the_next_larger(self):
        m = plan([[(1024 * k, 1024 * k, 100 * k, 0) for k in range(1, 80)]])
        self.assertEqual(len(m.TABLE), m.SIZES)
        self.assertEqual(m._own(79 * 1024, 79 * 1024), 7900 * M)


class Logprobs(unittest.TestCase):
    def test_logprob_rows_are_priced_past_a_plain_row(self):
        m = plan([PROBE])
        m._fold(1024, 1024, 1656 * M, 1024)
        m._refresh()
        self.assertEqual(m.LP, 1024 * M / 1024)
        self.assertEqual(m._own(1024, 1024, m.LP), 1656 * M)

    def test_logprob_forwards_alone_are_no_table(self):
        m = plan()
        m._fold(668, 668, 1900 * M, 668)
        m._refresh()
        self.assertFalse(m.TABLE)


class Carried(unittest.TestCase):
    def test_the_table_comes_back_as_it_went(self):
        m = plan([PROBE, R65, R131])
        m._fold(1024, 1024, 1656 * M, 1024)
        m._refresh()
        rec = m._table_out()
        n = plan()
        n._table_in(rec)
        self.assertEqual(n.TABLE, m.TABLE)
        self.assertEqual(n.LP, m.LP)
        for c, s in ((16384, 65536), (22016, 524288), (19456, 131072)):
            self.assertEqual(n._own(c, s), m._own(c, s))

    def test_a_forward_that_grew_more_raises_its_size(self):
        m = plan([R131])
        m._fold(22016, 66048, 9100 * M, 0)
        m._refresh()
        self.assertEqual(m._own(22016, 524288), 9100 * M)

    def test_a_forward_enters_without_the_scratch_kept(self):
        # the same forward in two launches: 7,364 MiB over 1,232 MiB kept, 6,689 MiB over 570 MiB (11-19)
        m = plan()
        m.observe_window(20224, 20224, 7364 * M, kept=1232 * M)
        m.observe_window(20224, 20224, 6689 * M, kept=570 * M)
        self.assertEqual([p[2] // M for p in m.PENDING], [6132, 6119])


class PlannedAgain(unittest.TestCase):
    # three forwards up to sequence 49,152, and the fourth, at 65,536, which came in after the next prompt's plan
    CAL = [(16384, 16384, 4846, 0), (16384, 32768, 5898, 0), (16384, 49152, 6920, 0), (1024, 1024, 590, 0)]

    def test_three_rising_forwards_price_a_small_chunk_past_ten_gib(self):
        m = plan([self.CAL])
        self.assertGreater(m._own(1024, 131072), 10000 * M)

    def test_a_plan_past_the_measured_sequence_is_made_again_once(self):
        m = plan([self.CAL])
        m.PLAN["r"], m.AHEAD["r"] = 3072, m.longest()
        self.assertFalse(m.rid_ahead("r", 128000))  # nothing further along yet
        m._fold(16384, 65536, 6277 * M, 0)
        m._refresh()
        self.assertLess(m._own(16384, 131072), 7000 * M)
        self.assertTrue(m.rid_ahead("r", 128000))
        self.assertFalse(m.rid_ahead("r", 128000))  # until a forward reaches further again

    def test_the_last_chunk_and_a_plan_within_the_measured_sequence_stay(self):
        m = plan([self.CAL])
        m.PLAN["r"], m.AHEAD["r"] = 3072, 1024
        self.assertFalse(m.rid_ahead("r", 3072))
        m.PLAN["q"] = 3072
        self.assertFalse(m.rid_ahead("q", 128000))


if __name__ == "__main__":
    unittest.main()
