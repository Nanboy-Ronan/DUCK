"""Decompose fairness gaps into tool, planning, and reasoning components.

Reads existing fairness outputs (per-question features + lens CSVs) and computes:
1) Raw disparity gap by attribute.
2) Planning-standardized gap (controls for trajectory/tool-selection buckets).
3) Planning+reasoning-standardized gap (adds reasoning-style buckets).

This is designed as a lightweight, log-only test to check whether component-level
bias proxies fully explain observed disparity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def find_fairness_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        fp = p / "fairness_posthoc"
        if fp.is_dir():
            out.append(fp)
    return out


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pick_tool_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("tool_used__")])


def to_int_flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(lower=0, upper=1)


def make_plan_bucket(df: pd.DataFrame) -> pd.Series:
    tool_cols = pick_tool_columns(df)
    first_tool = df["first_tool"].fillna("NONE").astype(str) if "first_tool" in df else pd.Series(["NONE"] * len(df))
    tcc = pd.to_numeric(df.get("tool_call_count", pd.Series([0] * len(df))), errors="coerce").fillna(0)

    # Coarse call-count bins to keep enough support per stratum.
    bins = pd.cut(tcc, bins=[-1, 0, 2, 5, 9999], labels=["0", "1-2", "3-5", "6+"])

    if not tool_cols:
        return first_tool.str.cat(bins.astype(str), sep="|calls=")

    used = pd.DataFrame({c: to_int_flag(df[c]) for c in tool_cols})
    tool_set = []
    for row in used.itertuples(index=False):
        active = [tool_cols[i].replace("tool_used__", "") for i, v in enumerate(row) if int(v) == 1]
        tool_set.append(",".join(active) if active else "NONE")
    tool_set_s = pd.Series(tool_set, index=df.index)
    return first_tool.str.cat(tool_set_s, sep="|set=").str.cat(bins.astype(str), sep="|calls=")


def qbucket(series: pd.Series, q: int = 4) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < 8 or x.nunique(dropna=True) <= 1:
        return pd.Series(["ALL"] * len(series), index=series.index)
    b = pd.qcut(x, q=q, duplicates="drop")
    return b.astype(str)


def make_reason_bucket(df: pd.DataFrame) -> pd.Series:
    hedge = qbucket(df.get("hedge_count", pd.Series([0] * len(df))))
    cert = qbucket(df.get("certainty_count", pd.Series([0] * len(df))))
    incons = qbucket(df.get("inconsistency_markers", pd.Series([0] * len(df))))
    refusal = to_int_flag(df.get("refusal_flag", pd.Series([0] * len(df)))).astype(str)
    return hedge.str.cat(cert, sep="|c=").str.cat(incons, sep="|i=").str.cat(refusal, sep="|r=")


def standardized_group_accuracy(df: pd.DataFrame, group_col: str, bucket_col: str) -> pd.DataFrame:
    # Global bucket distribution.
    w = df[bucket_col].value_counts(normalize=True, dropna=False).to_dict()

    rows = []
    for grp, gdf in df.groupby(group_col):
        gmean = float(pd.to_numeric(gdf["is_correct"], errors="coerce").mean())
        for b, wb in w.items():
            gb = gdf[gdf[bucket_col] == b]
            if len(gb) > 0:
                acc = float(pd.to_numeric(gb["is_correct"], errors="coerce").mean())
            else:
                acc = gmean
            rows.append((grp, b, wb, acc))
    tmp = pd.DataFrame(rows, columns=[group_col, "bucket", "weight", "acc"])
    out = (
        tmp.assign(weighted=lambda d: d["weight"] * d["acc"])
        .groupby(group_col, as_index=False)["weighted"]
        .sum()
        .rename(columns={"weighted": "std_acc"})
    )
    return out


def max_gap(df: pd.DataFrame, col: str) -> tuple[float, str, str]:
    d = df[[col, "group"]].dropna()
    if d.empty or len(d) < 2:
        return np.nan, "", ""
    lo = d.loc[d[col].idxmin()]
    hi = d.loc[d[col].idxmax()]
    return float(hi[col] - lo[col]), str(hi["group"]), str(lo["group"])


def lens_max_abs(df: pd.DataFrame, attr: str, col: str) -> float:
    if df.empty or col not in df.columns or "attribute" not in df.columns:
        return np.nan
    sub = df[df["attribute"] == attr]
    if sub.empty:
        return np.nan
    vals = pd.to_numeric(sub[col], errors="coerce").abs()
    return float(vals.max()) if vals.notna().any() else np.nan


def analyze_baseline(fp: Path, min_group_n: int) -> list[dict]:
    per_q = safe_read_csv(fp / "per_question_features.csv")
    if per_q.empty:
        return []

    per_q["is_correct"] = pd.to_numeric(per_q["is_correct"], errors="coerce")
    per_q = per_q[per_q["is_correct"].notna()].copy()
    if per_q.empty:
        return []

    inherited = safe_read_csv(fp / "lens_inherited_tool_bias.csv")
    planning = safe_read_csv(fp / "lens_agentic_tool_selection_bias.csv")
    reasoning = safe_read_csv(fp / "lens_llm_reasoning_bias.csv")

    out: list[dict] = []
    for attr in ("gender_norm", "age_group"):
        if attr not in per_q.columns:
            continue
        df = per_q[per_q[attr].notna()].copy()
        df[attr] = df[attr].astype(str)
        df = df[df[attr] != "UNKNOWN"]
        if df.empty:
            continue

        sizes = df[attr].value_counts()
        keep = sizes[sizes >= min_group_n].index
        df = df[df[attr].isin(keep)]
        if df[attr].nunique() < 2:
            continue

        raw = (
            df.groupby(attr, as_index=False)["is_correct"]
            .mean()
            .rename(columns={attr: "group", "is_correct": "acc"})
        )
        raw_gap, best_raw, worst_raw = max_gap(raw, "acc")

        df["plan_bucket"] = make_plan_bucket(df)
        plan_std = standardized_group_accuracy(df, attr, "plan_bucket").rename(columns={attr: "group"})
        plan_gap, _, _ = max_gap(plan_std.rename(columns={"std_acc": "acc"}), "acc")

        df["reason_bucket"] = make_reason_bucket(df)
        df["combo_bucket"] = df["plan_bucket"].astype(str) + "||" + df["reason_bucket"].astype(str)
        full_std = standardized_group_accuracy(df, attr, "combo_bucket").rename(columns={attr: "group"})
        full_gap, _, _ = max_gap(full_std.rename(columns={"std_acc": "acc"}), "acc")

        tool_fail_gap = lens_max_abs(inherited, attr, "tool_failure_gap")
        tool_harm_gap = lens_max_abs(inherited, attr, "min_acc_diff_used_minus_not")
        planning_gap_proxy = lens_max_abs(planning, attr, "usage_gap")
        reasoning_gap_proxy = lens_max_abs(reasoning, attr, "abs_gap")

        plan_explained = float(max(0.0, raw_gap - plan_gap)) if pd.notna(raw_gap) and pd.notna(plan_gap) else np.nan
        reason_explained = float(max(0.0, plan_gap - full_gap)) if pd.notna(plan_gap) and pd.notna(full_gap) else np.nan
        residual_gap = float(full_gap) if pd.notna(full_gap) else np.nan

        recs: list[str] = []
        if pd.notna(tool_fail_gap) and tool_fail_gap >= 0.02 or pd.notna(tool_harm_gap) and tool_harm_gap >= 0.08:
            recs.append("tool: add per-tool reliability gating + fallback chain for high-risk tools")
        if pd.notna(planning_gap_proxy) and planning_gap_proxy >= 0.08 or pd.notna(plan_explained) and plan_explained >= 0.01:
            recs.append("planning: apply demographic-blind planner prompt and first-tool policy constraints")
        if pd.notna(reasoning_gap_proxy) and reasoning_gap_proxy >= 0.05 or pd.notna(reason_explained) and reason_explained >= 0.01:
            recs.append("reasoning: enforce evidence-cited synthesis and contradiction check")
        if pd.notna(residual_gap) and residual_gap >= 0.02:
            recs.append("system: residual gap remains; run counterfactual rewrites and calibration by subgroup")
        if not recs:
            recs.append("monitor: no strong trigger; keep periodic fairness drift checks")

        out.append(
            {
                "baseline": fp.parent.name,
                "attribute": attr,
                "n": int(len(df)),
                "n_groups": int(df[attr].nunique()),
                "best_group_raw": best_raw,
                "worst_group_raw": worst_raw,
                "raw_gap": float(raw_gap),
                "planning_adjusted_gap": float(plan_gap),
                "planning_reasoning_adjusted_gap": float(full_gap),
                "plan_explained_gap": plan_explained,
                "reason_explained_gap": reason_explained,
                "residual_gap": residual_gap,
                "tool_failure_gap_max_abs": tool_fail_gap,
                "tool_harm_gap_max_abs": tool_harm_gap,
                "planning_usage_gap_max_abs": planning_gap_proxy,
                "reasoning_feature_gap_max_abs": reasoning_gap_proxy,
                "recommendations": "; ".join(recs),
            }
        )

    return out


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Bias Decomposition Summary")
    lines.append("")
    lines.append("Columns:")
    lines.append("- raw_gap: max group accuracy gap")
    lines.append("- planning_adjusted_gap: gap after standardizing for trajectory/tool-selection buckets")
    lines.append("- planning_reasoning_adjusted_gap: gap after additional reasoning-style control")
    lines.append("- residual_gap: remaining disparity after both controls")
    lines.append("")
    if df.empty:
        lines.append("No analyzable baselines found.")
    else:
        show_cols = [
            "baseline",
            "attribute",
            "raw_gap",
            "planning_adjusted_gap",
            "planning_reasoning_adjusted_gap",
            "residual_gap",
            "planning_usage_gap_max_abs",
            "reasoning_feature_gap_max_abs",
            "tool_harm_gap_max_abs",
        ]
        lines.append(df[show_cols].to_markdown(index=False))
        lines.append("")
        for _, r in df.iterrows():
            lines.append(f"## {r['baseline']} | {r['attribute']}")
            lines.append(f"- Worst vs best group (raw): `{r['worst_group_raw']}` vs `{r['best_group_raw']}`")
            lines.append(f"- Recommendations: {r['recommendations']}")
            lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        default="./logs/chexagentbench/analysis",
        help="Root with per-baseline fairness_posthoc outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default="./logs/chexagentbench/analysis/decomposition",
        help="Directory to write decomposition outputs.",
    )
    parser.add_argument(
        "--min-group-n",
        type=int,
        default=50,
        help="Minimum group size to include in disparity calculations.",
    )
    args = parser.parse_args()

    root = Path(args.analysis_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for fp in find_fairness_dirs(root):
        rows.extend(analyze_baseline(fp, args.min_group_n))

    res = pd.DataFrame(rows)
    csv_path = out_dir / "bias_decomposition.csv"
    json_path = out_dir / "bias_decomposition.json"
    md_path = out_dir / "bias_decomposition.md"

    if not res.empty:
        res = res.sort_values(["baseline", "attribute"]).reset_index(drop=True)

    res.to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump(res.to_dict(orient="records"), f, indent=2)
    write_markdown(res, md_path)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
