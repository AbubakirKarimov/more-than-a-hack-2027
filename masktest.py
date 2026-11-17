#!/usr/bin/env python3
"""
Unit tests for the pitch masks and helpers in model.py: penalty box, final
third, Left/Central/Right zones, segment merging, ball gap filling, half-time
split, kick-off formation, plus a synthetic end-to-end run.

  python masktest.py            # or: pytest masktest.py -q

If the real data is present (MTAH_DATA_DIR or ./data), one extra test loads it
and checks shapes.
"""
import os
import unittest

import numpy as np

import model as M

L, W = M.PITCH_L, M.PITCH_W


def arr(*v):
    return np.array(v, dtype=float)


class BoxMask(unittest.TestCase):
    def count(self, xs, ys, direction):
        px = np.array([xs], dtype=float)
        py = np.array([ys], dtype=float)
        return int(M.box_count(px, py, np.ones(len(xs), bool), np.array([direction]))[0])

    def test_attacking_right(self):
        self.assertEqual(self.count([100], [34], 1), 1)          # deep in the box
        self.assertEqual(self.count([105 - 16.4], [34], 1), 1)   # just inside the 16.5 line
        self.assertEqual(self.count([105 - 16.6], [34], 1), 0)   # just outside
        self.assertEqual(self.count([100], [34 + 20.0], 1), 1)   # inside the side line
        self.assertEqual(self.count([100], [34 + 20.4], 1), 0)   # outside the side line
        self.assertEqual(self.count([5], [34], 1), 0)            # own box does not count

    def test_attacking_left_mirrors(self):
        self.assertEqual(self.count([5], [34], -1), 1)
        self.assertEqual(self.count([16.6], [34], -1), 0)
        self.assertEqual(self.count([100], [34], -1), 0)

    def test_nan_and_no_direction(self):
        self.assertEqual(self.count([np.nan, 100], [34, 34], 1), 1)
        self.assertEqual(self.count([100], [34], 0), 0)

    def test_counts_many(self):
        self.assertEqual(self.count([95, 96, 97, 98, 99, 100, 101, 60], [34] * 8, 1), 7)


class FinalThird(unittest.TestCase):
    def test_both_directions(self):
        bx = arr(71, 69, 34, 36, 52.5)
        np.testing.assert_array_equal(M.in_final_third(bx, np.ones(5, int)), [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(M.in_final_third(bx, -np.ones(5, int)), [0, 0, 1, 0, 0])

    def test_nan_ball_is_false(self):
        self.assertFalse(M.in_final_third(arr(np.nan), np.array([1]))[0])

    def test_no_direction(self):
        self.assertFalse(M.in_final_third(arr(100), np.array([0]))[0])


class Zones(unittest.TestCase):
    def test_attacking_right_low_y_is_left(self):
        z = M.zone_of(arr(5, 34, 60), np.ones(3, int), W, left_is_low_y=True)
        self.assertEqual(list(z), ["Left", "Central", "Right"])

    def test_attacking_left_flips(self):
        z = M.zone_of(arr(5, 34, 60), -np.ones(3, int), W, left_is_low_y=True)
        self.assertEqual(list(z), ["Right", "Central", "Left"])

    def test_convention_switch(self):
        z = M.zone_of(arr(5), np.array([1]), W, left_is_low_y=False)
        self.assertEqual(z[0], "Right")

    def test_unknown(self):
        z = M.zone_of(arr(np.nan, 5), np.array([1, 0]), W)
        self.assertEqual(list(z), ["", ""])

    def test_boundaries(self):
        z = M.zone_of(arr(W / 3 - 0.01, W / 3 + 0.01, 2 * W / 3 + 0.01), np.ones(3, int), W)
        self.assertEqual(list(z), ["Left", "Central", "Right"])


class Segments(unittest.TestCase):
    def test_basic(self):
        m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
        self.assertEqual(M.segments(m), [(1, 2), (5, 5), (7, 9)])

    def test_merge_and_min_len(self):
        m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
        self.assertEqual(M.segments(m, merge_gap=1), [(1, 2), (5, 9)])
        self.assertEqual(M.segments(m, merge_gap=2), [(1, 9)])
        self.assertEqual(M.segments(m, min_len=3), [(7, 9)])

    def test_empty_and_full(self):
        self.assertEqual(M.segments(np.zeros(5, bool)), [])
        self.assertEqual(M.segments(np.ones(5, bool)), [(0, 4)])


class GapFill(unittest.TestCase):
    def test_short_gap_filled_long_kept(self):
        a = arr(0, np.nan, 2, np.nan, np.nan, np.nan, 6, np.nan)
        out = M.fill_short_gaps(a, max_gap=2)
        self.assertAlmostEqual(out[1], 1.0)
        self.assertTrue(np.isnan(out[3:6]).all())  # gap of 3 > 2
        self.assertTrue(np.isnan(out[7]))          # trailing edge never extrapolated


class HalfSplit(unittest.TestCase):
    def test_noisy_split(self):
        rng = np.random.default_rng(0)
        d = np.concatenate([np.full(600, 5.0), np.full(400, -5.0)]) + rng.normal(0, 3, 1000)
        self.assertLess(abs(M.find_split(d) - 600), 30)


def synthetic_match(fps=25, seconds=120, ko_at=10, goal_ko_at=70):
    """22 players + ball. Kick-off formation for ~2 s before ko_at and goal_ko_at;
    in between BAR (slots 0-10, attacking +x) pushes into ATH's box."""
    n = seconds * fps
    rng = np.random.default_rng(1)
    P = np.full((n, 22, 3), np.nan, np.float32)
    B = np.full((n, 3), np.nan, np.float32)
    left = np.column_stack([np.linspace(10, 50, 11), np.linspace(5, 63, 11)])
    right = np.column_stack([np.linspace(55, 95, 11), np.linspace(5, 63, 11)])
    left[-1] = (52.0, 34.0)   # BAR kicker on the spot
    for t in range(n):
        s = t / fps
        if (ko_at - 3 <= s <= ko_at) or (goal_ko_at - 3 <= s <= goal_ko_at):
            P[t, :11, :2] = left + rng.normal(0, .2, left.shape)
            P[t, 11:, :2] = right + rng.normal(0, .2, right.shape)
            B[t, :2] = (52.5, 34.0)
        elif 30 <= s <= 40:   # attack: ball at x=95, 4 BAR players in the ATH box
            P[t, :11, :2] = left + 40
            P[t, :4, :2] = [(92, 30), (94, 34), (96, 38), (98, 40)]
            P[t, 4:11, :2] = (60, 34)
            P[t, 11:, :2] = right
            B[t, :2] = (95, 20)
        else:                 # midfield play
            P[t, :11, :2] = left + 15
            P[t, 11:, :2] = right - 15
            B[t, :2] = (50 + 5 * np.sin(s), 34)
    P[:, :, 0] /= L; P[:, :, 1] /= W
    B[:, 0] /= L; B[:, 1] /= W
    meta = M.Meta(fps=fps, length=L, width=W, team_ids=["b", "a"],
                  team_names={"b": "Barcelona", "a": "Athletic Club"},
                  team_codes={"b": "BAR", "a": "ATH"},
                  slot_team=["b"] * 11 + ["a"] * 11, gk_slot={}, end_frame=n)
    return np.arange(1, n + 1), P, B, meta


class EndToEnd(unittest.TestCase):
    def test_synthetic(self):
        frames, P, B, meta = synthetic_match()
        prm = M.Params(orient_window_s=5, half_end_min_gap_s=1000, ko_cluster_s=20)
        fr, ev, info = M.detect(frames, P, B, meta, prm)
        self.assertEqual(len(fr), len(frames))
        ko = ev[ev.category == "Set Pieces"]
        self.assertEqual(len(ko), 2, ko)
        self.assertEqual(ko.iloc[0].subtype, M.KO_START)
        self.assertEqual(ko.iloc[1].subtype, M.KO_GOAL)
        self.assertAlmostEqual(ko.iloc[0].start_s, 10, delta=0.5)
        ft = ev[(ev.category == "Ball in Final Third") & (ev.team == "BAR")]
        self.assertEqual(len(ft), 1, ft)
        self.assertAlmostEqual(ft.iloc[0].start_s, 30, delta=0.5)
        self.assertAlmostEqual(ft.iloc[0].end_s, 40, delta=0.5)
        self.assertEqual(ft.iloc[0].zone, "Left")              # y=20 of 68, attacking +x
        self.assertEqual(ft.iloc[0].max_players_in_box, "4")
        pib = ev[ev.category == "Players in the Box"]
        self.assertEqual(len(pib), 1)
        self.assertEqual(pib.iloc[0].max_players_in_box, "4")
        self.assertFalse((ev.team == "ATH").any() and (ev.category == "Ball in Final Third").any())


class RealData(unittest.TestCase):
    def test_real_files_if_present(self):
        atd, xml = M.default_paths()
        if not (os.path.exists(atd) and os.path.exists(xml)):
            self.skipTest("match data not present")
        meta = M.load_meta(xml)
        self.assertEqual(len(meta.slot_team), 46)
        self.assertEqual([meta.team_codes[t] for t in meta.team_ids], ["BAR", "ATH"])
        frames, P, B = M.load_tracking(atd, 46)
        self.assertEqual(P.shape[1:], (46, 3))
        self.assertEqual(len(frames), meta.end_frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
