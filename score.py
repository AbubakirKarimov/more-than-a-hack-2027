#!/usr/bin/env python3
"""
Score predicted events (events.csv from model.py) against the ground truth
(BAR-ATH_tactical_smart_tagging.xml, or a CSV with the same columns as events.csv).

Matching is done per category:
  * Set Pieces          -> start times within --ko-tol seconds
  * other categories    -> temporal IoU >= --iou
Greedy one-to-one matching, best pairs first. Attributes (team, half, zone,
max players, subtype) are then scored on the matched pairs.

Usage:
  python score.py --pred out/events.csv --gt data/tactical/BAR-ATH_tactical_smart_tagging.xml
  python score.py --gt <xml> --dump          # show how the XML codes/labels were parsed
  python score.py ... --offset -0.04 --json out/score.json

This is a local scorer for tuning. The official number is whatever
evaluation/run_evaluation.py says.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd

CATS = ("Set Pieces", "Players in the Box", "Ball in Final Third")
COLS = ["category", "subtype", "team", "half", "start_s", "end_s", "zone", "max_players_in_box"]


# ----------------------------------------------------------------------------
# Normalisation (shared by GT and predictions)
# ----------------------------------------------------------------------------
def norm_category(text: str):
    t = text.lower()
    if "kick" in t or "set piece" in t:
        return "Set Pieces"
    if "box" in t and "player" in t:
        return "Players in the Box"
    if "third" in t:
        return "Ball in Final Third"
    return None


def norm_team(text: str) -> str:
    t = str(text).lower()
    if "barc" in t or t.strip() == "bar":
        return "BAR"
    if "athl" in t or "atl" in t or "bilbao" in t or t.strip() == "ath":
        return "ATH"
    return ""


def norm_subtype(text: str) -> str:
    t = str(text).lower()
    if "after goal" in t or "goal" in t:
        return "Kick Off (After Goal)"
    if "start" in t:
        return "Kick Off (Start)"
    return ""


def norm_zone(text: str) -> str:
    t = str(text).strip().lower()
    if t in ("left", "l"):
        return "Left"
    if t in ("central", "centre", "center", "middle", "c"):
        return "Central"
    if t in ("right", "r"):
        return "Right"
    return ""


def norm_players(text) -> str:
    m = re.search(r"(\d+)", str(text))
    if not m:
        return ""
    n = int(m.group(1))
    return "7+" if n >= 7 else str(n)


def norm_half(text) -> int:
    t = str(text).lower()
    if re.search(r"\b(1st|first)\b", t) or t.strip() in ("1", "1.0", "h1"):
        return 1
    if re.search(r"\b(2nd|second)\b", t) or t.strip() in ("2", "2.0", "h2"):
        return 2
    return 0


# ----------------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------------
def _strip_ns(root):
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def parse_tagging_xml(path: str, verbose=False) -> pd.DataFrame:
    """Sportscode-style XML: <instance><start/><end/><code/><label><group/><text/></label>...</instance>.
    Labels may also be <label><text/></label> without a group."""
    root = _strip_ns(ET.parse(path).getroot())
    rows, skipped = [], Counter()
    insts = [el for el in root.iter() if isinstance(el.tag, str) and el.tag.lower() == "instance"]
    for inst in insts:
        def txt(name):
            for ch in inst:
                if ch.tag.lower() == name:
                    return (ch.text or "").strip()
            return ""
        try:
            start, end = float(txt("start")), float(txt("end"))
        except ValueError:
            skipped["bad start/end"] += 1
            continue
        code = txt("code")
        labels = []
        for lab in inst:
            if lab.tag.lower() != "label":
                continue
            g = (lab.findtext("group") or "").strip()
            t = (lab.findtext("text") or "").strip()
            labels.append((g, t))
        cat = norm_category(code) or next((norm_category(t) for g, t in labels if norm_category(t)), None)
        if cat is None:
            skipped[code] += 1
            continue
        r = dict(category=cat, subtype="", team="", half=0, start_s=start, end_s=end,
                 zone="", max_players_in_box="", code=code,
                 labels="; ".join(f"{g}={t}" if g else t for g, t in labels))
        everything = [code] + [t for _, t in labels]
        for g, t in labels:
            gl = g.lower()
            if "team" in gl and norm_team(t):
                r["team"] = norm_team(t)
            elif "half" in gl and norm_half(t):
                r["half"] = norm_half(t)
            elif ("zone" in gl or "side" in gl or "lane" in gl or "channel" in gl) and norm_zone(t):
                r["zone"] = norm_zone(t)
            elif "player" in gl and norm_players(t):
                r["max_players_in_box"] = norm_players(t)
        # ungrouped labels / info inside code text
        for t in everything:
            r["team"] = r["team"] or norm_team(t)
            r["half"] = r["half"] or norm_half(t)
            r["zone"] = r["zone"] or norm_zone(t)
            if not r["max_players_in_box"]:
                m = re.search(r"max players in the box\D*(\d+)", t, re.I)
                if m:
                    r["max_players_in_box"] = norm_players(m.group(1))
        if cat == "Set Pieces":
            r["subtype"] = next((norm_subtype(t) for t in everything if norm_subtype(t)), "")
        rows.append(r)
    df = pd.DataFrame(rows)
    if verbose or skipped:
        print(f"[gt] {len(insts)} instances, {len(df)} kept, skipped: {dict(skipped)}", file=sys.stderr)
    return df


def load_events(path: str, verbose=False) -> pd.DataFrame:
    if path.lower().endswith(".xml"):
        df = parse_tagging_xml(path, verbose)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df["category"] = df["category"].map(lambda c: c if c in CATS else norm_category(c))
        df = df[df["category"].notna()].copy()
        for c, f in (("team", norm_team), ("zone", norm_zone), ("max_players_in_box", norm_players),
                     ("subtype", norm_subtype), ("half", norm_half)):
            if c in df:
                df[c] = df[c].map(f)
            else:
                df[c] = 0 if c == "half" else ""
        df["start_s"] = df["start_s"].astype(float)
        df["end_s"] = df["end_s"].astype(float)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------
def iou_matrix(p: pd.DataFrame, g: pd.DataFrame) -> np.ndarray:
    ps, pe = p["start_s"].to_numpy()[:, None], p["end_s"].to_numpy()[:, None]
    gs, ge = g["start_s"].to_numpy()[None, :], g["end_s"].to_numpy()[None, :]
    inter = np.clip(np.minimum(pe, ge) - np.maximum(ps, gs), 0, None)
    union = np.maximum(pe, ge) - np.minimum(ps, gs)
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def greedy_match(score: np.ndarray, thr: float):
    pairs = []
    if score.size == 0:
        return pairs
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
    used_p, used_g = set(), set()
    for i, j in order:
        if score[i, j] < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        pairs.append((int(i), int(j), float(score[i, j])))
    return pairs


def score(pred: pd.DataFrame, gt: pd.DataFrame, iou_thr=0.3, ko_tol=3.0):
    out = {"per_category": {}, "attributes": {}}
    f1s = []
    attr = {k: [] for k in ("team", "half", "zone", "max_players_in_box", "max_players_pm1", "subtype")}
    for cat in CATS:
        p = pred[pred["category"] == cat].reset_index(drop=True)
        g = gt[gt["category"] == cat].reset_index(drop=True)
        if cat == "Set Pieces":
            dt = np.abs(p["start_s"].to_numpy()[:, None] - g["start_s"].to_numpy()[None, :])
            sc = np.where(dt <= ko_tol, 1 - dt / (ko_tol + 1e-9) + 1e-6, 0.0)
            pairs = greedy_match(sc, 1e-9)
        else:
            pairs = greedy_match(iou_matrix(p, g), iou_thr)
        tp = len(pairs); fp = len(p) - tp; fn = len(g) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if len(g):
            f1s.append(f1)
        ious = [iou_matrix(p.iloc[[i]], g.iloc[[j]])[0, 0] for i, j, _ in pairs]
        out["per_category"][cat] = dict(gt=len(g), pred=len(p), tp=tp, fp=fp, fn=fn,
                                        precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
                                        mean_iou=round(float(np.mean(ious)), 4) if ious else None)
        for i, j, _ in pairs:
            pr, gr = p.iloc[i], g.iloc[j]
            for k in ("team", "half"):
                if gr[k] not in ("", 0):
                    attr[k].append(str(pr[k]) == str(gr[k]))
            if cat == "Ball in Final Third" and gr["zone"]:
                attr["zone"].append(pr["zone"] == gr["zone"])
            if cat != "Set Pieces" and gr["max_players_in_box"]:
                a = int(str(pr["max_players_in_box"]).rstrip("+") or -9)
                b = int(str(gr["max_players_in_box"]).rstrip("+"))
                attr["max_players_in_box"].append(a == b)
                attr["max_players_pm1"].append(abs(a - b) <= 1)
            if cat == "Set Pieces" and gr["subtype"]:
                attr["subtype"].append(pr["subtype"] == gr["subtype"])
    out["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
    out["attributes"] = {k: dict(n=len(v), accuracy=round(float(np.mean(v)), 4) if v else None)
                         for k, v in attr.items()}
    return out


def print_report(res):
    print(f"{'category':22s} {'gt':>5s} {'pred':>5s} {'tp':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'mIoU':>6s}")
    for cat, r in res["per_category"].items():
        miou = f"{r['mean_iou']:.3f}" if r["mean_iou"] is not None else "  -  "
        print(f"{cat:22s} {r['gt']:5d} {r['pred']:5d} {r['tp']:5d} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} {r['f1']:6.3f} {miou:>6s}")
    print(f"macro F1: {res['macro_f1']:.4f}")
    print("attribute accuracy on matched events:")
    for k, v in res["attributes"].items():
        acc = f"{v['accuracy']:.3f}" if v["accuracy"] is not None else "-"
        print(f"  {k:22s} n={v['n']:4d}  acc={acc}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="predicted events.csv")
    ap.add_argument("--gt", required=True, help="smart_tagging.xml or GT csv")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--ko-tol", type=float, default=3.0, help="kick-off start tolerance (s)")
    ap.add_argument("--offset", type=float, default=0.0, help="seconds added to predicted times")
    ap.add_argument("--json", help="write results JSON here")
    ap.add_argument("--dump", action="store_true", help="print how the GT was parsed and exit")
    a = ap.parse_args(argv)

    gt = load_events(a.gt, verbose=True)
    if a.dump:
        print(gt.groupby(["category", "subtype", "team", "half", "zone"], dropna=False).size().to_string())
        if "code" in gt:
            print("\ndistinct codes:", Counter(gt["code"]).most_common())
            print("\nsample labels:\n" + "\n".join(gt["labels"].head(10)))
        unk = {c: int((gt[c].isin(["", 0])).sum()) for c in ("team", "half", "zone", "max_players_in_box")}
        print("\nempty attribute counts:", unk)
        return
    if not a.pred:
        ap.error("--pred is required unless --dump")
    pred = load_events(a.pred)
    pred["start_s"] += a.offset
    pred["end_s"] += a.offset
    res = score(pred, gt, a.iou, a.ko_tol)
    print_report(res)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
