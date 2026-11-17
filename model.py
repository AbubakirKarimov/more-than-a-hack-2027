#!/usr/bin/env python3
"""
Rule-based tactical event detector for More Than A Hack 2027 (BAR-ATH).

Reads the Metrica ATD tracking (txt + xml) and produces:
  frames.csv   one row per tracking frame (row i = frame i)
  events.csv   detected events: Set Pieces (kick-offs), Players in the Box,
               Ball in Final Third, with team / half / zone / max players
  run_info.json  half-time split, half starts, kick-offs, parameters used

Usage:
  python model.py --atd data/tactical/BAR-ATH_tactical_atd.txt \
                  --xml data/tactical/BAR-ATH_tactical_atd.xml --out out/
  python model.py --params my_params.json      # override any Params field

Every function below works on plain numpy arrays so masktest.py can test it
without the real data.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import warnings
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Pitch geometry (metres). Overridden by FieldSize in the XML when present.
# ----------------------------------------------------------------------------
PITCH_L, PITCH_W = 105.0, 68.0
BOX_DEPTH = 16.5          # penalty area depth from goal line
BOX_HALF_W = 20.16        # half of 40.32 m penalty area width
THIRD_DEPTH = 35.0        # final third = last 35 m
BOX_BEHIND_LINE_TOL = 2.0 # tracking noise: allow players slightly past goal line

CATEGORIES = ("Set Pieces", "Players in the Box", "Ball in Final Third")
KO_START, KO_GOAL = "Kick Off (Start)", "Kick Off (After Goal)"
ZONES = ("Left", "Central", "Right")


@dataclass
class Params:
    ball_gap_fill_s: float = 2.0     # interpolate ball gaps up to this long
    min_visible_play: int = 10       # fewer visible players => not live play
    # Ball in Final Third
    ft_min_s: float = 1.0
    ft_merge_s: float = 1.0
    # Players in the Box
    pib_min_players: int = 1
    pib_min_s: float = 0.5
    pib_merge_s: float = 1.0
    pib_require_ball_ft: bool = True
    # Kick-offs: detected from the formation (the ball is usually untracked
    # while it sits on the spot): both teams in their own half, a player on the
    # centre spot, ball missing or near the spot.
    ko_kicker_radius_m: float = 4.0
    ko_ball_radius_m: float = 9.15   # centre circle
    ko_min_visible: int = 14
    ko_min_per_team: int = 5
    ko_own_half_frac: float = 0.9
    ko_own_half_tol_m: float = 1.0
    ko_min_frames: int = 10
    ko_merge_s: float = 2.0
    ko_cluster_s: float = 90.0
    ko_event_len_s: float = 2.0
    # Half-time detection
    orient_window_s: float = 60.0
    h2_search_back_s: float = 600.0
    half_end_min_visible: int = 16   # smoothed visible players below this = break / end
    half_end_window_s: float = 30.0
    half_end_min_gap_s: float = 180.0   # a break must last this long
    # Zone convention: when attacking towards +x, is the attacker's LEFT the
    # low-y side? True if y grows downwards in the video. Verify vs labels.
    left_is_low_y: bool = True

    @classmethod
    def from_json(cls, path: str) -> "Params":
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        bad = set(data) - known
        if bad:
            raise ValueError(f"unknown params: {sorted(bad)}")
        return cls(**data)


@dataclass
class Meta:
    fps: float
    length: float
    width: float
    team_ids: List[str]          # [t0, t1]; t0 attacks +x in the first half
    team_names: Dict[str, str]
    team_codes: Dict[str, str]
    slot_team: List[str]         # team id per tracking slot, in file order
    gk_slot: Dict[str, int]
    end_frame: int


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def team_code(name: str) -> str:
    n = name.lower()
    if "barc" in n:
        return "BAR"
    if "athl" in n or "atl" in n or "bilbao" in n:
        return "ATH"
    return name[:3].upper()


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def load_meta(xml_path: str) -> Meta:
    root = _strip_ns(ET.parse(xml_path).getroot())
    fps = float(root.findtext(".//FrameRate") or 25.0)
    length = float(root.findtext(".//FieldSize/Width") or PITCH_L)
    width = float(root.findtext(".//FieldSize/Height") or PITCH_W)

    names, dirs = {}, {}
    for t in root.iter("Team"):
        tid = t.get("id")
        names[tid] = (t.findtext("Name") or tid).strip()
        for pp in t.iter("ProviderParameter"):
            if pp.findtext("Name") == "attack_direction_first_half":
                dirs[tid] = 1 if (pp.findtext("Value") or "").startswith("left") else -1
    if len(names) != 2:
        raise ValueError(f"expected 2 teams in {xml_path}, found {len(names)}")
    tids = list(names)
    if dirs.get(tids[0], 1) != 1:
        tids = tids[::-1]
    if dirs and dirs.get(tids[0]) != 1:
        warnings.warn("no team listed as attacking left_to_right; using XML order")

    players = {p.get("id"): p for p in root.iter("Player")}
    order = []
    for ref in root.iter("PlayerChannelRef"):
        cid = ref.get("playerChannelId", "")
        if cid.endswith("_x"):
            order.append(cid[len("player"):-2])
    if not order:  # fall back to Player order
        order = list(players)
    slot_team, gk = [], {}
    for i, pid in enumerate(order):
        p = players[pid]
        tid = p.get("teamId")
        slot_team.append(tid)
        for pp in p.iter("ProviderParameter"):
            if pp.findtext("Name") == "position_type" and pp.findtext("Value") == "Goalkeeper":
                gk[tid] = i
    spec = root.find(".//DataFormatSpecification")
    end_frame = int(spec.get("endFrame")) if spec is not None and spec.get("endFrame") else -1
    return Meta(fps, length, width, tids, names,
                {t: team_code(names[t]) for t in tids}, slot_team, gk, end_frame)


def load_tracking(txt_path: str, n_slots: int, cache: str | None = None):
    """Return frames (N,), players (N, n_slots, 3), ball (N, 3). Normalised units."""
    if cache and os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(txt_path):
        z = np.load(cache)
        return z["frames"], z["players"], z["ball"]
    with open(txt_path, "rb") as f:
        raw = f.read().replace(b":", b",").replace(b";", b",")
    ncol = 1 + 3 * n_slots + 3
    df = pd.read_csv(io.BytesIO(raw), header=None, na_values=["NaN"],
                     dtype=np.float64, engine="c")
    if df.shape[1] != ncol:
        raise ValueError(f"{txt_path}: expected {ncol} columns, got {df.shape[1]}")
    arr = df.to_numpy()
    frames = arr[:, 0].astype(np.int64)
    players = arr[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3).astype(np.float32)
    ball = arr[:, 1 + 3 * n_slots:].astype(np.float32)
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.savez(cache, frames=frames, players=players, ball=ball)
    return frames, players, ball


# ----------------------------------------------------------------------------
# Small array helpers (unit-tested in masktest.py)
# ----------------------------------------------------------------------------
def segments(mask: np.ndarray, merge_gap: int = 0, min_len: int = 1) -> List[Tuple[int, int]]:
    """Runs of True as inclusive (start, end); merge runs separated by <= merge_gap
    False frames; drop runs shorter than min_len frames."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    out: List[List[int]] = []
    for s, e in zip(starts, ends):
        if out and s - out[-1][1] - 1 <= merge_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(int(s), int(e)) for s, e in out if e - s + 1 >= min_len]


def fill_short_gaps(a: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate interior NaN runs of length <= max_gap."""
    a = np.asarray(a, dtype=np.float64)
    out = a.copy()
    ok = ~np.isnan(a)
    if ok.sum() < 2:
        return out
    idx = np.arange(len(a))
    interp = np.interp(idx, idx[ok], a[ok])
    first, last = np.flatnonzero(ok)[[0, -1]]
    for s, e in segments(~ok):
        if s > first and e < last and e - s + 1 <= max_gap:
            out[s:e + 1] = interp[s:e + 1]
    return out


def find_split(d: np.ndarray) -> int:
    """Index where sign(d) changes from mostly + to mostly - (single change point)."""
    s = np.sign(np.nan_to_num(d))
    pos = np.concatenate(([0], np.cumsum(s > 0)))
    neg = np.concatenate(([0], np.cumsum(s < 0)))
    n = len(s)
    score = pos[1:n] + (neg[n] - neg[1:n])   # split at k: [0,k) positive, [k,n) negative
    return int(np.argmax(score)) + 1


def in_final_third(bx: np.ndarray, direction: np.ndarray, length: float = PITCH_L) -> np.ndarray:
    """Ball in the final third of the team attacking `direction` (+1 => +x, -1 => -x, 0 => none)."""
    c = length / 2
    with np.errstate(invalid="ignore"):
        return (direction != 0) & ((bx - c) * direction >= c - THIRD_DEPTH)


def box_count(px: np.ndarray, py: np.ndarray, team_mask: np.ndarray, direction: np.ndarray,
              length: float = PITCH_L, width: float = PITCH_W) -> np.ndarray:
    """Players of the team (slots in team_mask) inside the box they attack, per frame."""
    c = length / 2
    x = (px[:, team_mask] - c) * direction[:, None]
    y = np.abs(py[:, team_mask] - width / 2)
    with np.errstate(invalid="ignore"):
        inside = (x >= c - BOX_DEPTH) & (x <= c + BOX_BEHIND_LINE_TOL) & (y <= BOX_HALF_W)
    inside &= (direction != 0)[:, None]
    return inside.sum(axis=1)


def zone_of(by: np.ndarray, direction: np.ndarray, width: float = PITCH_W,
            left_is_low_y: bool = True) -> np.ndarray:
    """Left / Central / Right from the attacking team's point of view ('' if unknown)."""
    by = np.asarray(by, dtype=np.float64)
    idx = np.where(by < width / 3, 0, np.where(by > 2 * width / 3, 2, 1))
    flip = direction * (1 if left_is_low_y else -1)
    fwd = np.array(ZONES, dtype=object)
    rev = fwd[::-1]
    out = np.where(flip > 0, fwd[idx], np.where(flip < 0, rev[idx], ""))
    out[np.isnan(by)] = ""
    return out.astype(object)


def players_label(n: int) -> str:
    return "7+" if n >= 7 else str(int(n))


def detect_kickoffs(px, py, vis, bx, by, m0, m1, length, width, prm: Params, fps: float):
    """Return list of (frame_index, orientation). orientation +1: t0 on the left half
    (first-half layout), -1: t0 on the right (second-half layout)."""
    cx, cy = length / 2, width / 2
    with np.errstate(invalid="ignore"):
        bdist = np.hypot(bx - cx, by - cy)
        ball_ok = np.isnan(bdist) | (bdist <= prm.ko_ball_radius_m)
        kick = np.min(np.where(vis, np.hypot(px - cx, py - cy), np.inf), axis=1) <= prm.ko_kicker_radius_m
        nvis = vis.sum(1)
        n0 = (vis & m0).sum(1)
        n1 = (vis & m1).sum(1)
        tol = prm.ko_own_half_tol_m
        base = (ball_ok & kick & (nvis >= prm.ko_min_visible)
                & (n0 >= prm.ko_min_per_team) & (n1 >= prm.ko_min_per_team))
        res = []
        for orient in (1, -1):
            own0 = (px <= cx + tol) if orient == 1 else (px >= cx - tol)
            own1 = (px >= cx - tol) if orient == 1 else (px <= cx + tol)
            cnt = (vis & ((m0[None, :] & own0) | (m1[None, :] & own1))).sum(1)
            frac = cnt / np.maximum(nvis, 1)
            cand = base & (frac >= prm.ko_own_half_frac)
            for s, e in segments(cand, merge_gap=int(prm.ko_merge_s * fps), min_len=prm.ko_min_frames):
                res.append((e, orient))   # moment the ball is played: end of the hold
    res.sort()
    clustered: List[Tuple[int, int]] = []
    gap = int(prm.ko_cluster_s * fps)
    for k, o in res:
        if clustered and k - clustered[-1][0] <= gap:
            clustered[-1] = (k, o)        # keep the latest hold in a cluster
        else:
            clustered.append((k, o))
    return clustered


def half_ends(nvis, h1_start, h1_limit, h2_start, prm: Params, fps: float):
    """Exclusive end index of each half: start of the first sustained stretch
    (>= half_end_min_gap_s) where the smoothed visible-player count drops below
    match level (players leave the pitch / camera stops following play)."""
    win = max(1, int(prm.half_end_window_s * fps))
    sm = pd.Series(nvis).rolling(win, center=True, min_periods=1).median().to_numpy()
    gaps = segments(sm < prm.half_end_min_visible, 0, max(1, int(prm.half_end_min_gap_s * fps)))

    def first_gap(lo, hi):
        for s, e in gaps:
            if lo < s < hi:
                return s
        return hi

    return first_gap(h1_start, h1_limit), first_gap(h2_start, len(nvis))


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def detect(frames, players, ball, meta: Meta, prm: Params):
    fps, L, W = meta.fps, meta.length, meta.width
    n = len(frames)
    t0, t1 = meta.team_ids
    c0, c1 = meta.team_codes[t0], meta.team_codes[t1]
    slot_team = np.array(meta.slot_team)
    m0, m1 = slot_team == t0, slot_team == t1

    px = players[:, :, 0].astype(np.float64) * L
    py = players[:, :, 1].astype(np.float64) * W
    vis = ~np.isnan(px)
    nvis = vis.sum(1)
    bx_raw = ball[:, 0].astype(np.float64) * L
    by_raw = ball[:, 1].astype(np.float64) * W
    gap = int(round(prm.ball_gap_fill_s * fps))
    bx, by = fill_short_gaps(bx_raw, gap), fill_short_gaps(by_raw, gap)

    # --- half-time: which side each team occupies ------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.nanmean(np.where(m1, px, np.nan), axis=1) - np.nanmean(np.where(m0, px, np.nan), axis=1)
    win = max(1, int(prm.orient_window_s * fps))
    ds = pd.Series(d).rolling(win, center=True, min_periods=max(1, win // 4)).median().to_numpy()
    split = find_split(ds)

    kos = detect_kickoffs(px, py, vis, bx_raw, by_raw, m0, m1, L, W, prm, fps)
    h1 = [k for k, o in kos if o == 1 and k < split]
    h2 = [k for k, o in kos if o == -1 and k >= split - int(prm.h2_search_back_s * fps)]
    h1_start = h1[0] if h1 else 0
    h2_start = h2[0] if h2 else split
    h1_end, h2_end = half_ends(nvis, h1_start, min(split, h2_start), h2_start, prm, fps)
    half = np.zeros(n, dtype=np.int8)
    half[h1_start:h1_end] = 1
    half[h2_start:h2_end] = 2
    dir0 = np.where(half == 1, 1, np.where(half == 2, -1, 0))
    dir1 = -dir0

    # --- per-frame features ------------------------------------------------------
    live = (half > 0) & (nvis >= prm.min_visible_play) & ~np.isnan(bx)
    ft = {c0: live & in_final_third(bx, dir0, L), c1: live & in_final_third(bx, dir1, L)}
    box = {c0: box_count(px, py, m0, dir0, L, W), c1: box_count(px, py, m1, dir1, L, W)}
    zone = {c0: zone_of(by, dir0, W, prm.left_is_low_y), c1: zone_of(by, dir1, W, prm.left_is_low_y)}

    events = []

    def add(cat, sub, team, s, e, zn="", maxp=""):
        events.append(dict(category=cat, subtype=sub, team=team, half=int(half[s]),
                           start_frame=int(frames[s]), end_frame=int(frames[e]),
                           start_s=round((frames[s] - 1) / fps, 3),
                           end_s=round(frames[e] / fps, 3), zone=zn, max_players_in_box=maxp))

    # --- Set pieces: kick-offs ---------------------------------------------------
    ko_flag = np.zeros(n, dtype=np.int8)
    ko_len = int(prm.ko_event_len_s * fps)
    used_kos = []
    for k, o in kos:
        h = half[k]
        if h == 0 or (h == 1 and o != 1) or (h == 2 and o != -1):
            continue
        sub = KO_START if k in (h1_start, h2_start) else KO_GOAL
        # kicker = player nearest the ball (or the centre spot when the ball is untracked)
        kx, ky = (bx_raw[k], by_raw[k]) if not np.isnan(bx_raw[k]) else (L / 2, W / 2)
        dist = np.hypot(px[k] - kx, py[k] - ky)
        dist[~vis[k]] = np.inf
        j = int(np.argmin(dist))
        team = meta.team_codes[meta.slot_team[j]] if np.isfinite(dist[j]) else ""
        add("Set Pieces", sub, team, k, min(n - 1, k + ko_len))
        ko_flag[k] = 1
        used_kos.append(dict(frame=int(frames[k]), time_s=round((frames[k] - 1) / fps, 2),
                             half=int(h), subtype=sub, team=team))

    # --- Ball in Final Third & Players in the Box ---------------------------------
    for code in (c0, c1):
        for s, e in segments(ft[code], int(prm.ft_merge_s * fps), max(1, int(prm.ft_min_s * fps))):
            zs = [z for z in zone[code][s:e + 1] if z]
            zn = max(set(zs), key=zs.count) if zs else ""
            add("Ball in Final Third", "", code, s, e, zn, players_label(box[code][s:e + 1].max()))
        pmask = live & (box[code] >= prm.pib_min_players)
        if prm.pib_require_ball_ft:
            pmask &= ft[code]
        for s, e in segments(pmask, int(prm.pib_merge_s * fps), max(1, int(prm.pib_min_s * fps))):
            add("Players in the Box", "", code, s, e, "", players_label(box[code][s:e + 1].max()))

    ev = pd.DataFrame(events, columns=["category", "subtype", "team", "half", "start_frame",
                                       "end_frame", "start_s", "end_s", "zone", "max_players_in_box"])
    ev = ev.sort_values(["start_s", "category"]).reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(1, len(ev) + 1))

    ft_team = np.where(ft[c0], c0, np.where(ft[c1], c1, ""))
    fz = np.where(ft[c0], zone[c0], np.where(ft[c1], zone[c1], ""))
    fr = pd.DataFrame({
        "frame": frames, "time_s": np.round((frames - 1) / fps, 3), "half": half,
        "ball_x": np.round(bx, 3), "ball_y": np.round(by, 3),
        "ft_team": ft_team, "zone": fz,
        f"box_{c0}": box[c0], f"box_{c1}": box[c1], "kickoff": ko_flag,
    })
    info = dict(n_frames=int(n), fps=fps, teams={c0: meta.team_names[t0], c1: meta.team_names[t1]},
                first_half_attacks_pos_x=c0, split_frame=int(frames[split]),
                h1_start_frame=int(frames[h1_start]), h1_end_frame=int(frames[h1_end - 1]),
                h2_start_frame=int(frames[h2_start]), h2_end_frame=int(frames[h2_end - 1]),
                h1_start_s=round((frames[h1_start] - 1) / fps, 2),
                h2_start_s=round((frames[h2_start] - 1) / fps, 2),
                kickoffs=used_kos, all_kickoff_candidates=len(kos),
                event_counts=ev.groupby("category").size().to_dict(), params=asdict(prm))
    return fr, ev, info


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def default_paths():
    base = os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), "tactical")
    return (os.path.join(base, "BAR-ATH_tactical_atd.txt"),
            os.path.join(base, "BAR-ATH_tactical_atd.xml"))


def main(argv=None):
    atd, xml = default_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atd", default=atd, help="tracking txt")
    ap.add_argument("--xml", default=xml, help="tracking metadata xml")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--params", help="JSON file overriding Params fields")
    ap.add_argument("--cache", help="npz cache path (default: <out>/tracking_cache.npz)")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)

    prm = Params.from_json(a.params) if a.params else Params()
    meta = load_meta(a.xml)
    cache = None if a.no_cache else (a.cache or os.path.join(a.out, "tracking_cache.npz"))
    frames, players, ball = load_tracking(a.atd, len(meta.slot_team), cache)
    if meta.end_frame > 0 and frames[-1] != meta.end_frame:
        print(f"warning: last frame {frames[-1]} != XML endFrame {meta.end_frame}", file=sys.stderr)

    fr, ev, info = detect(frames, players, ball, meta, prm)
    os.makedirs(a.out, exist_ok=True)
    fr.to_csv(os.path.join(a.out, "frames.csv"), index=False)
    ev.to_csv(os.path.join(a.out, "events.csv"), index=False)
    with open(os.path.join(a.out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"frames: {info['n_frames']}  half-time split @ frame {info['split_frame']}")
    print(f"H1 starts {info['h1_start_s']}s, H2 starts {info['h2_start_s']}s")
    print(f"kick-offs: {[(k['time_s'], k['subtype'], k['team']) for k in info['kickoffs']]}")
    print("events:", info["event_counts"])
    print(f"wrote {a.out}/frames.csv, events.csv, run_info.json")


if __name__ == "__main__":
    main()
