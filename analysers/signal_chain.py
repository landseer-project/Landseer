#!/usr/bin/env python3
"""
Signal-Chain Interference (Same-Stage Order Swap) — AB/BA Pair View

For each row (combo) and each ordered pair (A before B) inside the SAME stage
(pre*, in*, post*, deploy*):
  - PRIME is the row where ONLY A and B are swapped in that same stage (B before A),
    with all other stages' tool orders identical.
  - Compare AB row vs BA row for BOTH tools' main metrics:
      * desirability-aligned delta Δ = (to - from) if higher_is_better else (from - to)
      * absolute-threshold labeling on |Δ| with t1/t2 (metrics in [0,1] typical):
            |Δ| < t1        -> equal
            t1 ≤ |Δ| < t2   -> pos_moderate / neg_moderate
            |Δ| ≥ t2        -> pos_severe   / neg_severe
  - Aggregate to "signal_interference":
      False            -> all equal
      Moderate True    -> ≥1 moderate, no severe (others equal)
      Severe True      -> ≥1 severe,   no moderate (others equal)
      Mixed True       -> both severe and moderate present
  - interference_kind across changed metrics:
      positive (all Δ>0), negative (all Δ<0), mixed (both signs), none (all equal)
      NOTE: If signal_interference == "False", force interference_kind = "none".
  - interference_causes lists non-equal labels per tool: "A:<metric>_<label>;B:<metric>_<label>"
  - Also record acc_test_clean delta and sign (better/worse/equal), independent of main metrics.

Outputs ONE file:
  - signal_chain_same_stage_pairs.csv
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional
import pandas as pd

# -----------------------------
# Config (aligned with your latest GI script)
# -----------------------------
DEFAULT_METRICS = {
    "acc_test_clean": {"aliases": ["acc_test_clean", "acc.clean", "acc_clean", "clean_acc", "acc_test"], "higher_is_better": True},
    "asr": {"aliases": ["asr", "attack_success_rate", "attack_sr"], "higher_is_better": False},
    "ood_auc": {"aliases": ["ood_auc", "auroc_ood", "ood.auroc", "auroc_ood_maxconf"], "higher_is_better": True},
    "pgd_acc": {"aliases": ["acc_robust", "acc_roboust", "robust_acc", "pgd_acc", "acc.pgd", "acc_roboust"], "higher_is_better": True},
    "carlini_robustness": {"aliases": ["carlini_robustness", "carlini_acc"], "higher_is_better": True},
    "fingerprinting": {"aliases": ["fingerprinting", "dataset_fingerprinting", "fp_score", "fingerprint_score"], "higher_is_better": True},
    # "privacy_epsilon": {"aliases": ["privacy_epsilon", "dp_epsilon", "epsilon", "eps"], "higher_is_better": False},
    "mia_auc": {"aliases": ["mia_auc", "membership_inference_auc", "mi_auc"], "higher_is_better": False},
    "watermark_accuracy": {"aliases": ["watermark_accuracy", "watermark_acc", "wm_acc"], "higher_is_better": True},
    "wmacc_badnets": {"aliases": ["wmacc_badnets"], "higher_is_better": True},
    "drop10_score": {"aliases": ["drop10_score"], "higher_is_better": True},
}

DEFAULT_TOOL_MAIN = {
    "pre-xgbod": "ood_auc",
    "in-trades": "pgd_acc",
    "post_neuronprune": "asr",
    "post-magnet": "carlini_robustness",
    "deploy_dataset_inference": "fingerprinting",
    "deploy_dp": "acc_test_clean",
    "in-teaching": "acc_test_clean",
    "in-dp": "mia_auc",
    "watermarking": "watermark_accuracy",
    "watermarkbn": "wmacc_badnets",
    "deploy_explainshap": "drop10_score",
}

HEURISTIC_TOOL_MAIN = [
    (re.compile(r"xgbod", re.I), "ood_auc"),
    (re.compile(r"trades", re.I), "pgd_acc"),
    (re.compile(r"magnet", re.I), "carlini_robustness"),
    (re.compile(r"neuronprune", re.I), "asr"),  # post_neuronprune
    (re.compile(r"\bdeploy[_-]?dp\b", re.I), "acc_test_clean"),
    (re.compile(r"\bdeploy[_-]?dataset[_-]?inference\b", re.I), "fingerprinting"),
    (re.compile(r"\bin[-_]?teaching\b", re.I), "acc_test_clean"),
    (re.compile(r"\bin[-_]?dp\b", re.I), "mia_auc"),
    (re.compile(r"\bwatermark(ing)?\b", re.I), "watermark_accuracy"),
    (re.compile(r"\bwatermarkbn\b", re.I), "wmacc_badnets"),
    (re.compile(r"\bdeploy[_-]?explainshap\b", re.I), "drop10_score"),
]

# Split ONLY on explicit delimiters; DO NOT split on spaces so multi-word tools remain intact
TOKEN_SPLIT_RE = re.compile(r"[+|;,/]+")
STAGE_KEYS = ["pre", "in", "post", "deploy"]

# -----------------------------
# Helpers
# -----------------------------
def _build_alias_map(metrics_cfg: Dict[str, Dict]) -> Dict[str, str]:
    amap = {}
    for canon, spec in metrics_cfg.items():
        for a in spec.get("aliases", []):
            amap[a.lower()] = canon
        amap[canon.lower()] = canon
    return amap

def _out_metric_name(m: str) -> str:
    return "acc_test" if m == "acc_test_clean" else m

def _norm_token(tok: str) -> str:
    t = tok.strip().lower()
    t = re.sub(r'[\[\]\"\'`]', '', t)
    t = re.sub(r"\s+", "_", t)  # normalize internal spaces to underscores
    if t.endswith("_noop"):
        t = "noop"
    fixes = {
        "post-magnet": "post_magnet",
        "intrades": "in-trades",
        "in_teaching": "in-teaching",
        "inteaching": "in-teaching",
        "in_dp": "in-dp",
        "indp": "in-dp",
        "deploy-dp": "deploy_dp",
        "deploy_dataset_inference": "deploy_dataset_inference",
        "watermark": "watermarking",
        "watermarkbn": "watermarkbn",
        "neuronprune": "post_neuronprune",
        "post-neuronprune": "post_neuronprune",
    }
    return fixes.get(t, t)

def resolve_metric_columns_soft(df: pd.DataFrame, metrics_cfg: Dict[str, Dict]) -> Dict[str, str]:
    """Return canonical metric -> actual column name in df (no error if some are missing)."""
    lower_cols = {c.lower(): c for c in df.columns}
    alias_map = _build_alias_map(metrics_cfg)
    resolved = {}
    for alias_lower, canon in alias_map.items():
        if alias_lower in lower_cols and canon not in resolved:
            resolved[canon] = lower_cols[alias_lower]
    return resolved

# def desirability_change(to_val: float, from_val: float, higher_is_better: bool) -> float:
#     return (to_val - from_val) if higher_is_better else (from_val - to_val)

def desirability_change(to_val: float, from_val: float, higher_is_better: bool) -> Optional[float]:
    if pd.isna(to_val) or pd.isna(from_val):
        return None

    if abs(from_val) < 1e-6:
        return None

    if higher_is_better:
        return (to_val - from_val) / abs(from_val)
    else:
        return (from_val - to_val) / abs(from_val)

def label_abs_threshold(delta: Optional[float], t1: float, t2: float) -> str:
    if delta is None or pd.isna(delta):
        return "equal"
    mag = abs(delta)
    if mag < t1:  # negligible
        return "equal"
    if mag < t2:
        return "pos_moderate" if delta > 0 else "neg_moderate"
    return "pos_severe" if delta > 0 else "neg_severe"

def severity_bucket(labels: List[str]) -> str:
    has_mod = any(l.endswith("moderate") for l in labels if l != "equal")
    has_sev = any(l.endswith("severe") for l in labels if l != "equal")
    if not (has_mod or has_sev):
        return "False"
    if has_mod and has_sev:
        return "Mixed True"
    if has_sev:
        return "Severe True"
    return "Moderate True"

def direction_kind_from_deltas(deltas: List[float]) -> str:
    if not deltas:
        return "none"
    pos = any(d > 0 for d in deltas)
    neg = any(d < 0 for d in deltas)
    if pos and neg:
        return "mixed"
    if pos:
        return "positive"
    if neg:
        return "negative"
    return "none"

def find_combo_col(df: pd.DataFrame) -> str:
    candidates = [
        "combination_id", "combination", "combo_id", "combo_index",
        "combo", "comb_id", "combo_idx", "comb", "combo_name"
    ]
    lc = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lc:
            return lc[c]
    return ""  # we'll fallback to row index when empty

# Build a canonical stage signature for a row
def stage_signature(stages: Dict[str, List[str]]) -> str:
    return "||".join("+".join(stages[k]) for k in STAGE_KEYS)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Same-stage order-swap interference with AB/BA pair signatures.")
    ap.add_argument("--input", required=True, help="Path to a single CSV log")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--t1", type=float, default=0.02, help="Negligible threshold (absolute delta)")
    ap.add_argument("--t2", type=float, default=0.05, help="Severe threshold (absolute delta)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)

    # Metric columns: soft resolution + -1 -> NaN handling later
    metric_cols = resolve_metric_columns_soft(df, DEFAULT_METRICS)
    resolved_metrics = list(metric_cols.keys())

    # Identify combo id column (optional)
    combo_col = find_combo_col(df)

    # Build stage lists per row
    stage_lists = []
    for _, row in df.iterrows():
        stages = {k: [] for k in STAGE_KEYS}
        for c in row.index:
            cl = c.lower()
            for stage in STAGE_KEYS:
                if cl.startswith(stage):
                    cell = "" if pd.isna(row[c]) else str(row[c]).strip()
                    if not cell:
                        continue
                    for tok in TOKEN_SPLIT_RE.split(cell):
                        if not tok:
                            continue
                        nt = _norm_token(tok)
                        if nt and nt != "noop":
                            stages[stage].append(nt)
        stage_lists.append(stages)
    df["_stage_lists"] = stage_lists
    df["_sig"] = [stage_signature(sl) for sl in stage_lists]

    # Map signature -> row index
    sig_to_idx = {}
    # If multiple rows share same signature, keep the first (consistent with earlier pairing logic)
    for i, s in enumerate(df["_sig"]):
        if s not in sig_to_idx:
            sig_to_idx[s] = i

    # Pull metrics per row (with -1 -> NaN)
    def get_metrics(row):
        vals = {}
        for canon, col in metric_cols.items():
            v = float(row[col])
            vals[canon] = float("nan") if v == -1 else v
        return vals

    metrics_by_row = [get_metrics(r) for _, r in df.iterrows()]

    # Utility: main metric for a tool (use defaults/heuristics)
    def main_metric_for(tool: str) -> Optional[str]:
        if tool in DEFAULT_TOOL_MAIN:
            return DEFAULT_TOOL_MAIN[tool]
        # Heuristic fallback
        for rx, m in HEURISTIC_TOOL_MAIN:
            if rx.search(tool):
                return m
        return None

    # Iterate pairs per row
    records = []
    for ridx, row in df.iterrows():
        stages_ab = row["_stage_lists"]
        sig_ab = row["_sig"]
        combo_id_ab = str(row[combo_col]) if combo_col else f"row_{ridx}"
        metrics_ab = metrics_by_row[ridx]

        for stage in STAGE_KEYS:
            seq = stages_ab[stage]
            n = len(seq)
            if n < 2:
                continue
            # all ordered pairs A before B (i<j)
            for i in range(n):
                for j in range(i+1, n):
                    A, B = seq[i], seq[j]

                    # Build the BA-swapped signature for this row
                    swapped = {k: list(v) for k, v in stages_ab.items()}
                    swapped[stage][i], swapped[stage][j] = swapped[stage][j], swapped[stage][i]
                    sig_ba = stage_signature(swapped)

                    if sig_ba not in sig_to_idx:
                        # no matching prime found; skip
                        continue

                    pidx = sig_to_idx[sig_ba]
                    combo_id_ba = str(df.iloc[pidx][combo_col]) if combo_col else f"row_{pidx}"
                    metrics_ba = metrics_by_row[pidx]

                    # Compute per-tool deltas & labels (absolute thresholds)
                    def compute(tool: str):
                        main = main_metric_for(tool)
                        if not main or main not in metrics_ab or main not in metrics_ba:
                            return None, "equal", main
                        a = metrics_ab[main]
                        b = metrics_ba[main]
                        if pd.isna(a) or pd.isna(b):
                            return None, "equal", main
                        hib = DEFAULT_METRICS.get(main, {}).get("higher_is_better", True)
                        delta = desirability_change(a, b, hib)
                        lab = label_abs_threshold(delta, args.t1, args.t2)
                        return delta, lab, main

                    dA, lA, mA = compute(A)
                    dB, lB, mB = compute(B)

                    # Acc-test delta (AB - BA) raw (not desirability; we just report sign and value)
                    if "acc_test_clean" in metrics_ab and "acc_test_clean" in metrics_ba:
                        a_acc = metrics_ab["acc_test_clean"]
                        b_acc = metrics_ba["acc_test_clean"]
                        if pd.isna(a_acc) or pd.isna(b_acc):
                            d_acc, acc_sign = None, "equal"
                        else:
                            d_acc = a_acc - b_acc
                            acc_sign = "better" if d_acc > 0 else ("worse" if d_acc < 0 else "equal")
                    else:
                        d_acc, acc_sign = None, "equal"

                    # Aggregate severity and kind
                    sig_label = severity_bucket([lA, lB])
                    if sig_label == "False":
                        kind = "none"
                    else:
                        changed_deltas = [d for d in [dA, dB] if d is not None and label_abs_threshold(d, args.t1, args.t2) != "equal"]
                        kind = direction_kind_from_deltas(changed_deltas)

                    # Causes text
                    causes = []
                    if lA != "equal" and mA:
                        causes.append(f"{A}:{_out_metric_name(mA)}_{lA}")
                    if lB != "equal" and mB:
                        causes.append(f"{B}:{_out_metric_name(mB)}_{lB}")

                    records.append({
                        # Where/what
                        "stage": stage,
                        "A_tool": A,
                        "A_main_metric": _out_metric_name(mA) if mA else "",
                        "B_tool": B,
                        "B_main_metric": _out_metric_name(mB) if mB else "",
                        "combo_index_ab": combo_id_ab,
                        "combo_index_ba": combo_id_ba,
                        "signature_ab": sig_ab,
                        "signature_ba": sig_ba,

                        # Deltas & labels per-tool (desirability-aligned)
                        "rel_delta_A": dA,
                        "label_A": lA,
                        "rel_delta_B": dB,
                        "label_B": lB,

                        # Aggregates
                        "signal_interference": sig_label,
                        "interference_kind": kind,
                        "interference_causes": ";".join(causes),

                        # Clean-acc regardless of main metrics
                        "abs_delta_acc_test": d_acc,
                        "acc_test_sign": acc_sign,
                    })

    out_path = os.path.join(args.outdir, "signal_chain_same_stage_pairs.csv")
    pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()






