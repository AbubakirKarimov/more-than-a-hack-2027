#!/usr/bin/env python3
"""
Sanity-check model.py output before scoring or submitting.

  python validate.py --frames out/frames.csv --events out/events.csv \
                     --xml data/tactical/BAR-ATH_tactical_atd.xml

Checks
  frames.csv : one row per tracking frame (count = XML endFrame or --n-frames),
               frames contiguous, half in {0,1,2}, coordinates on/near the pitch,
               box counts 0..11, kickoff 0/1, team/zone values valid,
               ft_team set only while half > 0
  events.csv : required columns, known categories/subtypes/teams/zones,
               half 1/2, 0 <= start <= end <= match length, max players valid,
               no overlapping events of the same category and team (warning)

Exit code 1 if any error, 0 otherwise (warnings do not fail).
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

CATS = {"Set Pieces", "Players in the Box", "Ball in Final Third"}
SUBTYPES = {"Kick Off (Start)", "Kick Off (After Goal)"}
TEAMS = {"BAR", "ATH"}
ZONES = {"Left", "Central", "Right"}
PLAYERS = {str(i) for i in range(7)} | {"7+"}
FRAME_COLS = ["frame", "time_s", "half", "ball_x", "ball_y", "ft_team", "zone", "kickoff"]
EVENT_COLS = ["event_id", "category", "subtype", "team", "half", "start_frame", "end_frame",
              "start_s", "end_s", "zone", "max_players_in_box"]


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def check(self, cond, msg, warn=False):
        if not cond:
            (self.warn if warn else self.err)(msg)
        return cond


def expected_frames(xml_path):
    root = ET.parse(xml_path).getroot()
    for el in root.iter():
        if el.tag.endswith("DataFormatSpecification") and el.get("endFrame"):
            return int(el.get("endFrame")) - int(el.get("startFrame", 1)) + 1, \
                float(next((e.text for e in root.iter() if e.tag.endswith("FrameRate")), 25))
    return None, 25.0


def validate_frames(path, n_expected, r: Report):
    df = pd.read_csv(path, keep_default_na=False, na_values=["", "NaN", "nan"],
                     dtype={"ft_team": str, "zone": str})
    df["ft_team"] = df["ft_team"].fillna("")
    df["zone"] = df["zone"].fillna("")
    missing = [c for c in FRAME_COLS if c not in df]
    if not r.check(not missing, f"frames: missing columns {missing}"):
        return df
    box_cols = [c for c in df if c.startswith("box_")]
    r.check(len(box_cols) == 2, f"frames: expected 2 box_<TEAM> columns, got {box_cols}")
    if n_expected is not None:
        r.check(len(df) == n_expected, f"frames: {len(df)} rows, expected {n_expected} (one per frame)")
    fr = df["frame"].to_numpy()
    r.check(np.all(np.diff(fr) == 1), "frames: frame column is not contiguous / increasing by 1")
    r.check(df["half"].isin([0, 1, 2]).all(), "frames: half outside {0,1,2}")
    for c, lo, hi in (("ball_x", -5, 110), ("ball_y", -5, 73)):
        v = df[c].dropna()
        r.check(((v >= lo) & (v <= hi)).all(), f"frames: {c} outside [{lo},{hi}] m "
                f"({int(((v < lo) | (v > hi)).sum())} rows)", warn=True)
    for c in box_cols:
        r.check(df[c].between(0, 11).all(), f"frames: {c} outside 0..11")
    r.check(df["kickoff"].isin([0, 1]).all(), "frames: kickoff not 0/1")
    r.check(df["ft_team"].isin(TEAMS | {""}).all(), f"frames: bad ft_team values {set(df['ft_team']) - TEAMS - {''}}")
    r.check(df["zone"].isin(ZONES | {""}).all(), f"frames: bad zone values {set(df['zone']) - ZONES - {''}}")
    r.check(((df["ft_team"] == "") | (df["half"] > 0)).all(), "frames: ft_team set outside a half")
    r.check(((df["ft_team"] == "") == (df["zone"] == "")).mean() > 0.99,
            "frames: zone and ft_team disagree on >1% of rows", warn=True)
    h = df["half"].to_numpy()
    r.check((h == 1).any() and (h == 2).any(), "frames: both halves should be present", warn=True)
    if (h == 1).any() and (h == 2).any():
        r.check(np.flatnonzero(h == 1).max() < np.flatnonzero(h == 2).min(), "frames: half 1 and half 2 overlap")
    return df


def validate_events(path, duration_s, r: Report):
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = [c for c in EVENT_COLS if c not in df]
    if not r.check(not missing, f"events: missing columns {missing}"):
        return df
    r.check(len(df) > 0, "events: file is empty", warn=True)
    st, en = df["start_s"].astype(float), df["end_s"].astype(float)
    bad_cat = set(df["category"]) - CATS
    r.check(not bad_cat, f"events: unknown categories {bad_cat}")
    sp = df["category"] == "Set Pieces"
    r.check(df.loc[sp, "subtype"].isin(SUBTYPES).all(), "events: Set Pieces with missing/unknown subtype")
    r.check((df.loc[~sp, "subtype"] == "").all(), "events: subtype set on non-set-piece events", warn=True)
    r.check(df["team"].isin(TEAMS).all(), f"events: bad team values {set(df['team']) - TEAMS}")
    r.check(df["half"].isin(["1", "2"]).all(), f"events: half must be 1 or 2, got {set(df['half']) - {'1', '2'}}")
    r.check(df["event_id"].is_unique, "events: event_id not unique")
    r.check((st >= 0).all(), "events: negative start_s")
    r.check((en >= st).all(), f"events: {int((en < st).sum())} events end before they start")
    if duration_s:
        r.check((en <= duration_s + 1).all(), f"events: end_s beyond match length {duration_s:.1f}s")
    ft = df["category"] == "Ball in Final Third"
    r.check(df.loc[ft, "zone"].isin(ZONES).all(),
            f"events: {int((~df.loc[ft, 'zone'].isin(ZONES)).sum())} final-third events without a valid zone",
            warn=True)
    nsp = ~sp
    r.check(df.loc[nsp, "max_players_in_box"].isin(PLAYERS).all(), "events: bad max_players_in_box values")
    pib = df["category"] == "Players in the Box"
    r.check((df.loc[pib, "max_players_in_box"] != "0").all(), "events: Players in the Box with 0 players")
    sfs, efs = df["start_frame"].astype(int), df["end_frame"].astype(int)
    r.check((efs >= sfs).all(), "events: end_frame < start_frame")
    for (cat, team), g in df.assign(s=st, e=en).groupby(["category", "team"]):
        g = g.sort_values("s")
        ov = int((g["s"].to_numpy()[1:] < g["e"].to_numpy()[:-1]).sum())
        r.check(ov == 0, f"events: {ov} overlapping '{cat}' events for {team}", warn=True)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="out/frames.csv")
    ap.add_argument("--events", default="out/events.csv")
    ap.add_argument("--xml", help="tracking metadata xml (gives expected frame count)")
    ap.add_argument("--n-frames", type=int, help="expected frame count (overrides --xml)")
    a = ap.parse_args(argv)

    n, fps = (expected_frames(a.xml) if a.xml else (None, 25.0))
    if a.n_frames:
        n = a.n_frames
    r = Report()
    fr = validate_frames(a.frames, n, r)
    dur = (len(fr) / fps) if len(fr) else None
    ev = validate_events(a.events, dur, r)

    print(f"frames.csv: {len(fr)} rows   events.csv: {len(ev)} rows")
    if "category" in ev:
        print(ev["category"].value_counts().to_string())
    for w in r.warnings:
        print("WARN ", w)
    for e in r.errors:
        print("ERROR", e)
    print("OK" if not r.errors else f"FAILED ({len(r.errors)} errors)")
    sys.exit(1 if r.errors else 0)


if __name__ == "__main__":
    main()
