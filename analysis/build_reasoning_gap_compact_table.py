#!/usr/bin/env python3
"""Build a compact LaTeX table of reasoning-bias subgroup gaps.

Output table layout:
- Rows: Attribute (Gender/Age) x LLM backbone
- Columns: metric gap (abs_gap) for each dataset
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    root: Path


DATASETS = [
    DatasetSpec("chex", "CheXagentBench", Path("./logs/chexagentbench/analysis")),
    DatasetSpec("mimic", "MIMIC-CXR-QA", Path("./logs/mimic/analysis/fairness")),
]

ATTR_ORDER = ["gender_norm", "age_group"]
ATTR_LABELS = {"gender_norm": "Gender", "age_group": "Age"}

METRIC_ORDER_DEFAULT = [
    "reasoning_quality_score",
]

METRIC_ORDER_FALLBACK = [
    "hedge_count",
    "demographic_terms",
]

METRIC_SHORT = {
    "reasoning_quality_score": r"$\Delta$JudgeGap$\downarrow$",
    "hedge_count": r"$\Delta$Hedge$\downarrow$",
    "inconsistency_markers": r"$\Delta$Incons.$\downarrow$",
    "demographic_terms": r"$\Delta$Demo.$\downarrow$",
}

BACKBONE_ORDER = [
    "llama-3.1-8b-vllm",
    "llama-3.1-8b-vllm-mimic",
    "ministral-3-8b-vllm",
    "ministral-3-8b-vllm-mimic",
    "qwen3vl8b-vllm",
    "qwen3vl8b-vllm-mimic",
    "qwen38b-vllm",
    "qwen38b-vllm-mimic",
    "agent_gemini-3-flash-preview",
    "agent-gemini3flash-mimic",
]

BACKBONE_LABEL = {
    "llama-3.1-8b-vllm": "LLaMA3.1-8B",
    "llama-3.1-8b-vllm-mimic": "LLaMA3.1-8B",
    "ministral-3-8b-vllm": "Ministral-3-8B",
    "ministral-3-8b-vllm-mimic": "Ministral-3-8B",
    "qwen3vl8b-vllm": "Qwen3VL-8B",
    "qwen3vl8b-vllm-mimic": "Qwen3VL-8B",
    "qwen38b-vllm": "Qwen3-8B",
    "qwen38b-vllm-mimic": "Qwen3-8B",
    "agent_gemini-3-flash-preview": "Gemini3-Flash",
    "agent-gemini3flash-mimic": "Gemini3-Flash",
}

MODEL_ROWS = ["LLaMA3.1-8B", "Ministral-3-8B", "Qwen3VL-8B", "Qwen3-8B", "Gemini3-Flash"]


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in text)


def _bootstrap_abs_gap_stats(
    grouped: dict[str, np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float | pd._libs.missing.NAType, float | pd._libs.missing.NAType, float | pd._libs.missing.NAType]:
    arrays = [v for v in grouped.values() if len(v) > 0]
    if len(arrays) < 2:
        return pd.NA, pd.NA, pd.NA
    gaps: list[float] = []
    for _ in range(max(1, int(n_bootstrap))):
        means = []
        for arr in arrays:
            idx = rng.integers(0, len(arr), size=len(arr))
            means.append(float(np.mean(arr[idx])))
        if not means:
            continue
        gaps.append(float(abs(max(means) - min(means))))
    if not gaps:
        return pd.NA, pd.NA, pd.NA
    gaps_arr = np.asarray(gaps, dtype=float)
    std = float(np.std(gaps_arr, ddof=1 if len(gaps_arr) > 1 else 0))
    ci_low = float(np.quantile(gaps_arr, 0.025))
    ci_high = float(np.quantile(gaps_arr, 0.975))
    return std, ci_low, ci_high


def _abs_gap_from_groups(grouped: dict[str, np.ndarray]) -> float | pd._libs.missing.NAType:
    arrays = [v for v in grouped.values() if len(v) > 0]
    if len(arrays) < 2:
        return pd.NA
    means = [float(np.mean(arr)) for arr in arrays]
    if not means:
        return pd.NA
    return float(abs(max(means) - min(means)))


def _recompute_bootstrap_std(
    features_path: Path,
    attrs: list[str],
    metrics: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[
    tuple[str, str],
    tuple[
        float | pd._libs.missing.NAType,
        float | pd._libs.missing.NAType,
        float | pd._libs.missing.NAType,
        float | pd._libs.missing.NAType,
    ],
]:
    out: dict[
        tuple[str, str],
        tuple[
            float | pd._libs.missing.NAType,
            float | pd._libs.missing.NAType,
            float | pd._libs.missing.NAType,
            float | pd._libs.missing.NAType,
        ],
    ] = {}
    if not features_path.exists():
        return out
    use_cols = [c for c in set(attrs + metrics) if c]
    try:
        feat = pd.read_csv(features_path, usecols=lambda c: c in use_cols)
    except Exception:
        return out
    rng = np.random.default_rng(seed)
    for attr in attrs:
        if attr not in feat.columns:
            for metric in metrics:
                out[(attr, metric)] = (pd.NA, pd.NA, pd.NA, pd.NA)
            continue
        attr_series = feat[attr].astype(str).str.strip()
        for metric in metrics:
            if metric not in feat.columns:
                out[(attr, metric)] = (pd.NA, pd.NA, pd.NA, pd.NA)
                continue
            metric_vals = pd.to_numeric(feat[metric], errors="coerce")
            valid = (~metric_vals.isna()) & attr_series.ne("") & attr_series.ne("nan")
            clean_vals = metric_vals[valid]
            sub = pd.DataFrame({"attr": attr_series[valid], "metric": clean_vals})
            if sub.empty:
                out[(attr, metric)] = (pd.NA, pd.NA, pd.NA, pd.NA)
                continue
            grouped = {
                g: vals.to_numpy(dtype=float)
                for g, vals in sub.groupby("attr", dropna=False)["metric"]
            }
            abs_gap = _abs_gap_from_groups(grouped)
            std, ci_low, ci_high = _bootstrap_abs_gap_stats(grouped, n_bootstrap, rng)
            out[(attr, metric)] = (abs_gap, std, ci_low, ci_high)
    return out


def collect(n_bootstrap: int, bootstrap_seed: int) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for fp in sorted(ds.root.glob("*/fairness_posthoc/lens_llm_reasoning_bias.csv")):
            backbone_key = fp.parts[-3]
            df = pd.read_csv(fp)
            if df.empty:
                continue
            valid_metrics = METRIC_ORDER_DEFAULT + METRIC_ORDER_FALLBACK
            keep = df[df["attribute"].isin(ATTR_ORDER) & df["metric"].isin(valid_metrics)].copy()
            if keep.empty:
                continue
            keep["dataset"] = ds.label
            keep["dataset_key"] = ds.key
            keep["backbone_key"] = backbone_key
            keep["model"] = BACKBONE_LABEL.get(backbone_key, backbone_key)
            keep["abs_gap"] = pd.to_numeric(keep["abs_gap"], errors="coerce")
            features_path = fp.parent / "per_question_features.csv"
            valid_metrics = sorted(set(keep["metric"].astype(str).tolist()))
            model_seed = bootstrap_seed + sum(ord(ch) for ch in f"{ds.key}:{backbone_key}")
            recomputed = _recompute_bootstrap_std(
                features_path=features_path,
                attrs=ATTR_ORDER,
                metrics=valid_metrics,
                n_bootstrap=n_bootstrap,
                seed=model_seed,
            )
            keep["bootstrap_std"] = keep.apply(
                lambda r: recomputed.get((str(r["attribute"]), str(r["metric"])), (pd.NA, pd.NA, pd.NA, pd.NA))[1],
                axis=1,
            )
            keep["ci_low"] = keep.apply(
                lambda r: recomputed.get((str(r["attribute"]), str(r["metric"])), (pd.NA, pd.NA, pd.NA, pd.NA))[2],
                axis=1,
            )
            keep["ci_high"] = keep.apply(
                lambda r: recomputed.get((str(r["attribute"]), str(r["metric"])), (pd.NA, pd.NA, pd.NA, pd.NA))[3],
                axis=1,
            )
            keep["abs_gap"] = keep.apply(
                lambda r: recomputed.get((str(r["attribute"]), str(r["metric"])), (pd.NA, pd.NA, pd.NA, pd.NA))[0],
                axis=1,
            )
            rows.append(
                keep[
                    [
                        "dataset",
                        "dataset_key",
                        "backbone_key",
                        "model",
                        "attribute",
                        "metric",
                        "abs_gap",
                        "bootstrap_std",
                        "ci_low",
                        "ci_high",
                    ]
                ]
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "dataset_key",
                "backbone_key",
                "model",
                "attribute",
                "metric",
                "abs_gap",
                "bootstrap_std",
                "ci_low",
                "ci_high",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    return out


def metric_order_for(df: pd.DataFrame) -> list[str]:
    available = set(df["metric"].astype(str).tolist())
    preferred = METRIC_ORDER_DEFAULT + METRIC_ORDER_FALLBACK
    return [m for m in preferred if m in available]


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    metric_order = metric_order_for(df)
    # enforce ordering
    backbone_rank = {k: i for i, k in enumerate(BACKBONE_ORDER)}
    attr_rank = {k: i for i, k in enumerate(ATTR_ORDER)}
    metric_rank = {k: i for i, k in enumerate(metric_order)}
    out = df.copy()
    out["_b"] = out["backbone_key"].map(backbone_rank).fillna(999).astype(int)
    out["_a"] = out["attribute"].map(attr_rank).fillna(999).astype(int)
    out["_m"] = out["metric"].map(metric_rank).fillna(999).astype(int)
    out = out.sort_values(["_a", "_b", "_m"]).drop(columns=["_b", "_a", "_m"])
    return out


def value_or_dash(df: pd.DataFrame, dataset: str, model: str, attr: str, metric: str) -> str:
    sub = df[
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (df["attribute"] == attr)
        & (df["metric"] == metric)
    ]
    if sub.empty:
        return "--"
    v = sub["abs_gap"].iloc[0]
    lo = sub["ci_low"].iloc[0] if "ci_low" in sub.columns else pd.NA
    hi = sub["ci_high"].iloc[0] if "ci_high" in sub.columns else pd.NA
    if pd.isna(v):
        return "--"
    scale = 100.0
    if pd.isna(lo) or pd.isna(hi):
        return f"{float(v) * scale:.2f}"
    return f"${float(v) * scale:.2f}_{{[{float(lo) * scale:.2f},{float(hi) * scale:.2f}]}}$"


def render_table(df: pd.DataFrame, n_bootstrap: int) -> str:
    chex = DATASETS[0].label
    mimic = DATASETS[1].label
    metric_order = metric_order_for(df)
    if not metric_order:
        raise SystemExit("No supported reasoning metrics found.")
    n_metric = len(metric_order)
    tab_cols = "ll" + ("c" * (n_metric * 2))

    lines = []
    lines.append(r"\documentclass[10pt]{article}")
    lines.append(r"\usepackage[margin=0.8in]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{multirow}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{\textbf{Reasoning-bias subgroup gaps across two datasets.} "
        r"Lower is better for all $\Delta$ metrics. Unit: \%. "
        rf"Values are reported as mean subgroup gap with bootstrap 95\% CI ({n_bootstrap} non-parametric resamples).}}"
    )
    lines.append(r"    \label{tab:reasoning-bias}")
    lines.append(r"    \resizebox{\linewidth}{!}{")
    lines.append(f"    \\begin{{tabular}}{{{tab_cols}}}")
    lines.append(r"    \toprule")
    lines.append(
        f"    & & \\multicolumn{{{n_metric}}}{{c}}{{{latex_escape(chex)}}} & \\multicolumn{{{n_metric}}}{{c}}{{{latex_escape(mimic)}}} \\\\"
    )
    lines.append(r"    \midrule")
    metric_hdr = " & ".join([METRIC_SHORT[m] for m in metric_order])
    lines.append(f"    Attribute & LLM & {metric_hdr} & {metric_hdr}\\\\")
    lines.append(r"    \midrule")

    for attr in ATTR_ORDER:
        attr_label = ATTR_LABELS[attr]
        for i, model in enumerate(MODEL_ROWS):
            prefix = f"\\multirow{{{len(MODEL_ROWS)}}}{{*}}{{{attr_label}}} & " if i == 0 else "& "
            row_vals = []
            for ds in [chex, mimic]:
                for metric in metric_order:
                    row_vals.append(value_or_dash(df, ds, model, attr, metric))
            row_txt = " & ".join(row_vals)
            lines.append(f"    {prefix}\\textbf{{{model}}} & {row_txt} \\\\")
        if attr != ATTR_ORDER[-1]:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    }")
    lines.append(r"\end{table}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-tex",
        default="./logs/reasoning_bias_gap_compact_table.tex",
        help="Standalone LaTeX table output path.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100,
        help="Number of bootstrap resamples used to estimate std in this table.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for table bootstrap recomputation.",
    )
    args = parser.parse_args()

    df = collect(
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    if df.empty:
        raise SystemExit("No reasoning-bias files found.")

    ordered = build_matrix(df)

    tex = render_table(ordered, n_bootstrap=args.bootstrap_samples)
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex)

    counts = ordered.groupby("dataset")["model"].nunique().to_dict()
    print(f"[saved] tex: {out_tex}")
    print(f"[summary] models per dataset: {counts}")


if __name__ == "__main__":
    main()
