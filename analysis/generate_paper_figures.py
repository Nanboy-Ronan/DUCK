#!/usr/bin/env python3
"""Generate the paper figures from DUCX fairness analysis outputs.

This script consumes the per-model `fairness_posthoc` directories produced from
agent logs and writes only paper-figure artifacts plus their backing CSV data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TOOL_LABELS = {
    "chest_xray_classifier": "CLS",
    "classifier": "CLS",
    "chest_xray_expert": "QA",
    "llava_med_qa": "QA",
    "xray_vqa": "QA",
    "chest_xray_report_generator": "RG",
    "report_generator": "RG",
    "chest_xray_segmentation": "SEG",
    "segmentation": "SEG",
    "image_visualizer": "VIS",
    "visualizer": "VIS",
    "xray_phrase_grounding": "GRD",
    "phrase_grounding": "GRD",
    "START": "START",
}
TOOL_ORDER = ["SEG", "RG", "QA", "GRD", "CLS", "VIS"]
MATRIX_TOOLS = ["CLS", "QA", "RG", "SEG", "VIS", "GRD"]
MATRIX_FROM = ["START", *MATRIX_TOOLS]
MODEL_LABELS = {
    "agent-gemini3flash-preview": "Gemini3",
    "agent-gemini3flash-preview-mimic": "Gemini3",
    "llama-3.1-8b-vllm": "LLaMA3.1",
    "llama-3.1-8b-vllm-mimic": "LLaMA3.1",
    "ministral-3-8b-vllm": "Ministral-3",
    "ministral-3-8b-vllm-mimic": "Ministral-3",
    "qwen38b-vllm": "Qwen3",
    "qwen38b-vllm-mimic": "Qwen3",
    "qwen3vl8b-vllm": "Qwen3VL",
    "qwen3vl8b-vllm-mimic": "Qwen3VL",
}


def tool_label(name: object) -> str:
    raw = str(name or "").strip()
    return TOOL_LABELS.get(raw, raw.upper())


def model_label(path: Path) -> str:
    return MODEL_LABELS.get(path.parent.parent.name, path.parent.parent.name)


def find_posthoc_dirs(root: Path) -> List[Path]:
    dirs = sorted(p.parent for p in root.rglob("lens_agentic_transition_rates_by_group.csv"))
    return [p for p in dirs if p.name == "fairness_posthoc" and p.parent.name != "subsample"]


def load_tool_present_csv(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"baseline", "tool", "subgroup", "accuracy_when_tool_present"}
    missing = sorted(req - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df.copy()
    df["dataset"] = dataset
    df["model"] = df["baseline"].map(lambda x: MODEL_LABELS.get(str(x), str(x)))
    df["tool"] = df["tool"].map(tool_label)
    df["subgroup"] = df["subgroup"].astype(str).str.lower()
    df["accuracy_when_tool_present"] = pd.to_numeric(df["accuracy_when_tool_present"], errors="coerce")
    return df[df["tool"].isin(TOOL_ORDER)]


def build_figure2_data(tool_present: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = {
        "gender": ("female", "male"),
        "age": ("young", "old"),
    }
    for (dataset, model, tool), group in tool_present.groupby(["dataset", "model", "tool"]):
        for attribute, (a, b) in pairs.items():
            ga = group[group["subgroup"] == a]
            gb = group[group["subgroup"] == b]
            if ga.empty or gb.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "attribute": attribute,
                    "tool": tool,
                    "delta_acc_abs": abs(
                        float(ga["accuracy_when_tool_present"].iloc[0])
                        - float(gb["accuracy_when_tool_present"].iloc[0])
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "attribute", "tool", "model"])


def plot_figure2(fig2: pd.DataFrame, out_path: Path) -> None:
    panels = [
        ("CheXAgentBench", "gender", "(a) Gender on CheXAgentBench"),
        ("CheXAgentBench", "age", "(b) Age on CheXAgentBench"),
        ("MIMIC-FairnessVQA", "gender", "(c) Gender on MIMIC-FairnessVQA"),
        ("MIMIC-FairnessVQA", "age", "(d) Age on MIMIC-FairnessVQA"),
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6), squeeze=False)
    for ax, (dataset, attr, title) in zip(axes.ravel(), panels):
        sub = fig2[(fig2["dataset"] == dataset) & (fig2["attribute"] == attr)]
        data = [sub[sub["tool"] == tool]["delta_acc_abs"].dropna().to_numpy(float) for tool in TOOL_ORDER]
        vp = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, widths=0.72)
        for i, (body, vals) in enumerate(zip(vp["bodies"], data)):
            body.set_facecolor(colors[i % len(colors)])
            body.set_edgecolor("black")
            body.set_alpha(0.75)
            body.set_linewidth(1.2)
            if len(vals) == 0:
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            low = max(np.min(vals), q1 - 1.5 * iqr)
            high = min(np.max(vals), q3 + 1.5 * iqr)
            x = i + 1
            ax.vlines(x, low, high, color="#333333", lw=1.5, zorder=2)
            ax.vlines(x, q1, q3, color="#333333", lw=6, zorder=3)
            ax.scatter(x, med, marker="o", color="white", s=32, zorder=4, edgecolors="#333333", linewidths=0.4)
        ax.set_xticks(range(1, len(TOOL_ORDER) + 1))
        ax.set_xticklabels(TOOL_ORDER, rotation=25, ha="right", fontweight="bold")
        ax.set_ylabel(r"$\Delta$ ACC", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=320, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def load_transition_data(posthoc_dirs: Iterable[Tuple[str, Path]]) -> pd.DataFrame:
    frames = []
    for dataset, posthoc in posthoc_dirs:
        path = posthoc / "lens_agentic_transition_rates_by_group.csv"
        df = pd.read_csv(path)
        df["dataset"] = dataset
        df["model"] = model_label(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    split = df["transition"].astype(str).str.split("->", n=1, expand=True)
    df["from_tool"] = split[0].map(tool_label)
    df["to_tool"] = split[1].map(tool_label)
    df["group"] = df["group"].astype(str).str.lower()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)
    df = df[df["from_tool"].isin(MATRIX_FROM) & df["to_tool"].isin(MATRIX_TOOLS)]
    totals = df.groupby(["dataset", "model", "attribute", "group", "from_tool"])["count"].transform("sum")
    df["row_rate"] = np.where(totals > 0, df["count"] / totals, 0.0)
    return df


def build_figure3_data(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = {
        "gender_norm": ("male", "female", "gender"),
        "age_group": ("young", "old", "age"),
    }
    for (dataset, model, attr, from_tool, to_tool), group in transitions.groupby(
        ["dataset", "model", "attribute", "from_tool", "to_tool"]
    ):
        if attr not in pairs:
            continue
        positive, negative, attr_label = pairs[attr]
        p = group[group["group"] == positive]
        n = group[group["group"] == negative]
        if p.empty or n.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "attribute": attr_label,
                "from_tool": from_tool,
                "to_tool": to_tool,
                "delta_transition_prob": float(p["row_rate"].iloc[0]) - float(n["row_rate"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.groupby(["dataset", "attribute", "from_tool", "to_tool"], as_index=False)["delta_transition_prob"]
        .mean()
        .sort_values(["dataset", "attribute", "from_tool", "to_tool"])
    )


def plot_figure3(fig3: pd.DataFrame, out_path: Path) -> None:
    panels = [
        ("CheXAgentBench", "gender", r"(a) CheXAgentBench: $P_{male}-P_{female}$"),
        ("CheXAgentBench", "age", r"(b) CheXAgentBench: $P_{young}-P_{old}$"),
        ("MIMIC-FairnessVQA", "gender", r"(c) MIMIC-FairnessVQA: $P_{male}-P_{female}$"),
        ("MIMIC-FairnessVQA", "age", r"(d) MIMIC-FairnessVQA: $P_{young}-P_{old}$"),
    ]
    vmax = max(float(np.nanmax(np.abs(fig3["delta_transition_prob"].to_numpy(float)))), 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), squeeze=False)
    fig.subplots_adjust(right=0.87, wspace=0.28, hspace=0.40)

    for ax, (dataset, attr, title) in zip(axes.ravel(), panels):
        sub = fig3[(fig3["dataset"] == dataset) & (fig3["attribute"] == attr)]
        mat = (
            sub.pivot_table(index="from_tool", columns="to_tool", values="delta_transition_prob", aggfunc="mean")
            .reindex(index=MATRIX_FROM, columns=MATRIX_TOOLS)
            .fillna(0.0)
        )
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 12, "weight": "bold"},
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            xticklabels=MATRIX_TOOLS,
            yticklabels=MATRIX_FROM,
        )
        ax.set_xticklabels(MATRIX_TOOLS, rotation=0, fontweight="bold", fontsize=13)
        ax.set_yticklabels(MATRIX_FROM, rotation=0, fontweight="bold", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title, fontweight="bold", fontsize=14, pad=10)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.895, 0.14, 0.018, 0.72])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_ylabel("Transition-probability difference", rotation=90, labelpad=14, fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(out_path, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chex-analysis-root", type=Path, required=True)
    parser.add_argument("--mimic-analysis-root", type=Path, required=True)
    parser.add_argument("--chex-tool-present-csv", type=Path, required=True)
    parser.add_argument("--mimic-tool-present-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tool_present = pd.concat(
        [
            load_tool_present_csv(args.chex_tool_present_csv, "CheXAgentBench"),
            load_tool_present_csv(args.mimic_tool_present_csv, "MIMIC-FairnessVQA"),
        ],
        ignore_index=True,
    )
    fig2 = build_figure2_data(tool_present)
    fig2.to_csv(args.out_dir / "paper_figure2_tool_exposure_data.csv", index=False)
    plot_figure2(fig2, fig_dir / "paper_figure2_tool_exposure_bias.png")

    posthoc_dirs = [
        *[("CheXAgentBench", p) for p in find_posthoc_dirs(args.chex_analysis_root)],
        *[("MIMIC-FairnessVQA", p) for p in find_posthoc_dirs(args.mimic_analysis_root)],
    ]
    transitions = load_transition_data(posthoc_dirs)
    fig3 = build_figure3_data(transitions)
    fig3.to_csv(args.out_dir / "paper_figure3_tool_transition_data.csv", index=False)
    plot_figure3(fig3, fig_dir / "paper_figure3_tool_transition_bias.png")

    manifest = {
        "mode": "paper_figures",
        "figures": [
            "figures/paper_figure2_tool_exposure_bias.png",
            "figures/paper_figure3_tool_transition_bias.png",
        ],
        "data": [
            "paper_figure2_tool_exposure_data.csv",
            "paper_figure3_tool_transition_data.csv",
        ],
        "posthoc_dirs": [str(p) for _, p in posthoc_dirs],
    }
    (args.out_dir / "paper_figure_manifest.json").write_text(json.dumps(manifest, indent=2))

    allowed = {
        args.out_dir / "paper_figure2_tool_exposure_data.csv",
        args.out_dir / "paper_figure3_tool_transition_data.csv",
        args.out_dir / "paper_figure_manifest.json",
        fig_dir,
    }
    for path in args.out_dir.iterdir():
        if path not in allowed:
            if path.is_file():
                path.unlink()
    allowed_figs = {"paper_figure2_tool_exposure_bias.png", "paper_figure3_tool_transition_bias.png"}
    for path in fig_dir.iterdir():
        if path.name not in allowed_figs:
            path.unlink()


if __name__ == "__main__":
    main()
