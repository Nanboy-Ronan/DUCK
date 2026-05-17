#!/usr/bin/env python3
"""Post-hoc fairness analysis using agent logs only.

Reads log JSONL, joins question_id -> case_id and sensitive attributes,
extracts tool usage + LLM text features, and writes per-question dataset,
aggregate tables, and a concise markdown report.

python analysis/fairness_posthoc.py \
    --input-root ./logs/chexagentbench \
    --out-root ./logs/chexagentbench/analysis
"""
from __future__ import annotations

import json
import os
import re
import time
from ast import literal_eval
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI

LOG_PATH = Path("./logs/run.jsonl")
META_Q_PATH = Path("./data/chestagentbench/metadata.jsonl")
META_CASE_PATH = Path("./data/eurorad_metadata.json")
OUT_DIR = Path("./analysis/fairness")

RNG = np.random.default_rng(42)

HEDGE_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bpossibly\b",
    r"\bprobable\b",
    r"\blikely\b",
    r"\bcould\b",
    r"\buncertain\b",
    r"\bsuggests?\b",
    r"\bappears?\b",
]
CERT_PATTERNS = [
    r"\bdefinitely\b",
    r"\bclearly\b",
    r"\bcertainly\b",
    r"\bconsistent with\b",
    r"\bno evidence\b",
    r"\bmust be\b",
]
REFUSAL_PATTERNS = [
    r"\bI cannot\b",
    r"\bI'm unable\b",
    r"\bI can't\b",
    r"\bas an AI\b",
    r"\bnot able to\b",
]
DEMOGRAPHIC_PATTERNS = [
    r"\bmale\b",
    r"\bfemale\b",
    r"\bman\b",
    r"\bwoman\b",
    r"\bboy\b",
    r"\bgirl\b",
    r"\belderly\b",
    r"\byoung\b",
]
INCONSISTENCY_PATTERNS = [
    r"\bhowever\b",
    r"\bbut\b",
    r"\balthough\b",
]

PROMPT_DEMOGRAPHIC_PATTERNS = [
    r"\bmale\b",
    r"\bfemale\b",
    r"\bman\b",
    r"\bwoman\b",
    r"\bboy\b",
    r"\bgirl\b",
    r"\belderly\b",
    r"\byoung\b",
    r"\bpregnan\w*\b",
    r"\bold\b",
    r"\byear-old\b",
]

TOOL_FAIL_PATTERNS = [
    r"error",
    r"exception",
    r"traceback",
    r"failed",
    r"failure",
    r"not found",
    r"cannot",
    r"unable",
]

MIN_GROUP_SUPPORT = 10
MIN_BUCKET_SUPPORT = 8
BOOTSTRAP_SAMPLES = 1000
JUDGE_SCORE_MIN = 1.0
JUDGE_SCORE_MAX = 10.0
JUDGE_MAX_TEXT_CHARS = 2500


@dataclass
class ToolCall:
    name: str
    args: dict
    call_id: str | None


def _clip_text(s: Any, max_chars: int = JUDGE_MAX_TEXT_CHARS) -> str:
    text = str(s or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _extract_first_json_object(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _load_judge_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = str(row.get("cache_key", "")).strip()
            if key:
                cache[key] = row
    return cache


def _append_judge_cache_row(cache_path: Path, row: Dict[str, Any]) -> None:
    with cache_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _judge_reasoning_once(
    client: OpenAI,
    model: str,
    question_text: str,
    final_response: str,
    temperature: float = 0.0,
    timeout_sec: float = 120.0,
) -> Dict[str, Any]:
    system_prompt = (
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant "
        "to the user question displayed below. Your evaluation should consider factors such as the helpfulness, "
        "relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by "
        "providing a short explanation. Be as objective as possible. After providing your explanation, please rate "
        "the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: "
        "\"Rating: [[5]]\"."
    )
    user_prompt = (
        "[Question]\n"
        f"{_clip_text(question_text)}\n\n"
        "[The Start of Assistant’s Answer]\n"
        f"{_clip_text(final_response)}\n"
        "[The End of Assistant’s Answer]"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        timeout=timeout_sec,
    )
    content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    # Paper-style output parser: expects a rating like [[7]]
    m = re.search(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", content)
    if not m:
        # Fallback: accept "Rating: 7" style if model misses brackets.
        m = re.search(r"rating\s*[:=]?\s*(\d+(?:\.\d+)?)", content, flags=re.IGNORECASE)
    if not m:
        raise ValueError("judge response missing [[rating]]")
    score = pd.to_numeric(m.group(1), errors="coerce")
    if pd.isna(score):
        raise ValueError("judge rating is not numeric")
    score_f = float(np.clip(float(score), JUDGE_SCORE_MIN, JUDGE_SCORE_MAX))
    return {
        "reasoning_quality_score": score_f,
        "judge_confidence": np.nan,
        "judge_brief_rationale": "",
        "judge_raw": content[:4000],
    }


def score_reasoning_with_deepseek(
    df: pd.DataFrame,
    out_dir: Path,
    model: str,
    base_url: str,
    api_key_env: str,
    max_samples: Optional[int] = None,
    concurrency: int = 10,
    request_timeout_sec: float = 120.0,
    retries: int = 3,
    sleep_sec: float = 0.4,
) -> pd.DataFrame:
    key = (os.getenv(api_key_env) or "").strip()
    if not key:
        print(f"[judge] skip: env {api_key_env} is not set")
        df["reasoning_quality_score"] = np.nan
        df["judge_confidence"] = np.nan
        df["judge_model"] = ""
        df["judge_error"] = f"missing_api_key_env:{api_key_env}"
        return df

    cache_path = out_dir / "llm_judge_cache.jsonl"
    cache = _load_judge_cache(cache_path)
    out = df.copy()
    out["reasoning_quality_score"] = np.nan
    out["judge_confidence"] = np.nan
    out["judge_model"] = model
    out["judge_error"] = ""

    eligible = out[out["final_response_no_think"].fillna("").str.strip() != ""].copy()
    if max_samples is not None and max_samples > 0 and len(eligible) > max_samples:
        eligible = eligible.sample(n=max_samples, random_state=42)
    idxs = eligible.index.tolist()
    print(f"[judge] scoring rows={len(idxs)} model={model} base_url={base_url} concurrency={max(1, int(concurrency))}")

    pending: List[Dict[str, Any]] = []
    for idx in idxs:
        qid = str(out.at[idx, "question_id"])
        cache_key = f"{qid}::{model}::{base_url}"
        if cache_key in cache:
            row = cache[cache_key]
            out.at[idx, "reasoning_quality_score"] = pd.to_numeric(row.get("reasoning_quality_score"), errors="coerce")
            out.at[idx, "judge_confidence"] = pd.to_numeric(row.get("judge_confidence"), errors="coerce")
            out.at[idx, "judge_error"] = str(row.get("judge_error", ""))
        else:
            pending.append(
                {
                    "idx": idx,
                    "question_id": qid,
                    "cache_key": cache_key,
                    "question_text": str(out.at[idx, "question_text"]),
                    "final_response_no_think": str(out.at[idx, "final_response_no_think"]),
                }
            )

    def _judge_worker(item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        idx_local = int(item["idx"])
        qid_local = str(item["question_id"])
        cache_key_local = str(item["cache_key"])
        q_text = str(item["question_text"])
        final_resp = str(item["final_response_no_think"])

        client = OpenAI(api_key=key, base_url=base_url)
        last_err = ""
        result: Dict[str, Any] = {}
        for attempt in range(1, retries + 1):
            try:
                result = _judge_reasoning_once(
                    client,
                    model=model,
                    question_text=q_text,
                    final_response=final_resp,
                    timeout_sec=request_timeout_sec,
                )
                last_err = ""
                break
            except Exception as e:
                last_err = str(e)
                time.sleep(sleep_sec * attempt)

        row = {
            "cache_key": cache_key_local,
            "question_id": qid_local,
            "model": model,
            "base_url": base_url,
            "reasoning_quality_score": result.get("reasoning_quality_score", np.nan),
            "judge_confidence": result.get("judge_confidence", np.nan),
            "judge_brief_rationale": result.get("judge_brief_rationale", ""),
            "judge_raw": result.get("judge_raw", ""),
            "judge_error": last_err,
        }
        return idx_local, row

    total_pending = len(pending)
    if total_pending > 0:
        max_workers = max(1, int(concurrency))
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_judge_worker, item) for item in pending]
            for fut in as_completed(futures):
                idx_local, cache_row = fut.result()
                _append_judge_cache_row(cache_path, cache_row)
                cache[str(cache_row["cache_key"])] = cache_row

                out.at[idx_local, "reasoning_quality_score"] = pd.to_numeric(cache_row.get("reasoning_quality_score"), errors="coerce")
                out.at[idx_local, "judge_confidence"] = pd.to_numeric(cache_row.get("judge_confidence"), errors="coerce")
                out.at[idx_local, "judge_error"] = str(cache_row.get("judge_error", ""))

                done += 1
                if done % 50 == 0 or done == total_pending:
                    scored = int(out["reasoning_quality_score"].notna().sum())
                    print(f"[judge] progress {done}/{total_pending} scored={scored}")

    judged_cols = [
        "question_id",
        "reasoning_quality_score",
        "judge_confidence",
        "judge_model",
        "judge_error",
    ]
    out[judged_cols].to_csv(out_dir / "llm_reasoning_judge_scores.csv", index=False)
    return out


def read_metadata_question(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def read_metadata_case(path: Path) -> Dict[str, Dict[str, Any]]:
    # Supports JSON (dict/list) and CSV (e.g., MIMIC demographics table).
    out: Dict[str, Dict[str, Any]] = {}
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "case_id" in df.columns:
            id_col = "case_id"
        elif "dicom_id" in df.columns:
            id_col = "dicom_id"
        elif "id" in df.columns:
            id_col = "id"
        else:
            id_col = df.columns[0]
        for _, row in df.iterrows():
            cid = str(row.get(id_col))
            if not cid or cid.lower() in {"nan", "none", "null"}:
                continue
            out[cid] = row.to_dict()
        return out

    with path.open() as f:
        data = json.load(f)
    # normalize to dict by string case_id
    if isinstance(data, dict):
        for k, v in data.items():
            cid = str(v.get("case_id", k))
            out[cid] = v
    elif isinstance(data, list):
        for v in data:
            cid = str(v.get("case_id"))
            out[cid] = v
    return out


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def count_patterns(text: str, patterns: List[str]) -> int:
    if not text:
        return 0
    total = 0
    for p in patterns:
        total += len(re.findall(p, text, flags=re.IGNORECASE))
    return total


def tool_output_failed(output: str) -> bool:
    if not output:
        return False
    for p in TOOL_FAIL_PATTERNS:
        if re.search(p, output, flags=re.IGNORECASE):
            return True
    return False


def extract_text_fragments(content: Any) -> List[str]:
    """Extract text from heterogeneous message content payloads."""
    out: List[str] = []
    if content is None:
        return out
    if isinstance(content, str):
        s = content.strip()
        if s:
            out.append(s)
        return out
    if isinstance(content, list):
        for item in content:
            out.extend(extract_text_fragments(item))
        return out
    if isinstance(content, dict):
        # Common block schema: {"type": "text", "text": "..."}
        t = content.get("text")
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
        # Fallback for nested content fields.
        nested = content.get("content")
        if nested is not None:
            out.extend(extract_text_fragments(nested))
        return out
    return out


def parse_trace(trace: List[Dict[str, Any]]) -> Tuple[List[ToolCall], List[Tuple[str, str]], List[str]]:
    tool_calls: List[ToolCall] = []
    tool_outputs: List[Tuple[str, str]] = []
    ai_texts: List[str] = []
    for entry in trace or []:
        etype = entry.get("type")
        if etype == "ai":
            ai_texts.extend(extract_text_fragments(entry.get("content")))
            if entry.get("tool_calls"):
                for tc in entry.get("tool_calls"):
                    tool_calls.append(
                        ToolCall(
                            name=tc.get("name"),
                            args=tc.get("args") or {},
                            call_id=tc.get("id"),
                        )
                    )
        elif etype == "tool":
            name = entry.get("name")
            content = entry.get("content")
            if content is None:
                content = ""
            tool_outputs.append((name, str(content)))
        else:
            # include text if present
            ai_texts.extend(extract_text_fragments(entry.get("content")))
    return tool_calls, tool_outputs, ai_texts


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    lo, hi, _ = bootstrap_summary(values, n_boot=n_boot, alpha=alpha)
    return (lo, hi)


def bootstrap_summary(values: np.ndarray, n_boot: int = BOOTSTRAP_SAMPLES, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Non-parametric bootstrap over sample means: returns (ci_low, ci_high, bootstrap_std)."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]), 0.0)
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[idx].mean(axis=1)
    lo = np.quantile(samples, alpha / 2)
    hi = np.quantile(samples, 1 - alpha / 2)
    std = np.std(samples, ddof=1) if len(samples) > 1 else 0.0
    return (float(lo), float(hi), float(std))


def bootstrap_multigroup_gap_summary(
    values_by_group: Dict[str, np.ndarray],
    n_boot: int = BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Bootstrap summary for multigroup gap=max(group_mean)-min(group_mean)."""
    clean: Dict[str, np.ndarray] = {}
    for g, vals in values_by_group.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            clean[g] = arr
    if len(clean) < 2:
        return (np.nan, np.nan, np.nan)

    group_boot_means = []
    for arr in clean.values():
        if len(arr) == 1:
            group_boot_means.append(np.full(shape=(n_boot,), fill_value=float(arr[0]), dtype=float))
            continue
        idx = RNG.integers(0, len(arr), size=(n_boot, len(arr)))
        group_boot_means.append(arr[idx].mean(axis=1))

    boot_mat = np.vstack(group_boot_means)
    boot_gap = boot_mat.max(axis=0) - boot_mat.min(axis=0)
    lo = np.quantile(boot_gap, alpha / 2)
    hi = np.quantile(boot_gap, 1 - alpha / 2)
    std = np.std(boot_gap, ddof=1) if len(boot_gap) > 1 else 0.0
    return (float(lo), float(hi), float(std))


def safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def parse_list_like(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if "," in s and not (s.startswith("[") and s.endswith("]")):
            return [x.strip() for x in s.split(",") if x.strip()]
        if s.startswith("[") and s.endswith("]"):
            try:
                lit = literal_eval(s)
                if isinstance(lit, list):
                    return [str(v) for v in lit]
            except Exception:
                pass
        return [s]
    return [str(value)]


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def bh_fdr(p_values: List[float]) -> List[float]:
    if not p_values:
        return []
    n = len(p_values)
    order = np.argsort(p_values)
    ordered = np.array([p_values[i] for i in order], dtype=float)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = (ordered[i] * n) / rank
        prev = min(prev, val)
        adjusted[i] = min(prev, 1.0)
    out = np.empty(n, dtype=float)
    for idx, orig in enumerate(order):
        out[orig] = adjusted[idx]
    return [float(x) for x in out]


def save_bar_plot(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, hue: str | None = None, rotation: int = 20) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    if hue and hue in df.columns:
        piv = df.pivot(index=x, columns=hue, values=y).fillna(0.0)
        piv.plot(kind="bar")
        plt.ylabel(y)
    else:
        plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_matrix_plot(matrix_df: pd.DataFrame, title: str, out_path: Path) -> None:
    if matrix_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix_df.to_numpy(), aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Rate", rotation=90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_grouped_bar_from_pivot(pivot_df: pd.DataFrame, title: str, y_label: str, out_path: Path, rotation: int = 30) -> None:
    if pivot_df.empty:
        return
    ax = pivot_df.plot(kind="bar", figsize=(11, 5))
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


PAPER_TOOL_LABELS = {
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
PAPER_TOOL_ORDER = ["CLS", "QA", "RG", "SEG", "VIS", "GRD"]
PAPER_TRANSITION_ORDER = ["START", "CLS", "QA", "RG", "SEG", "VIS", "GRD"]


def paper_tool_label(name: Any) -> str:
    raw = str(name or "").strip()
    return PAPER_TOOL_LABELS.get(raw, raw.upper())


def infer_dataset_label(meta_q_path: Path, log_path: Path) -> str:
    text = f"{meta_q_path} {log_path}".lower()
    if "mimic" in text:
        return "MIMIC-FairnessVQA"
    return "CheXAgentBench"


def infer_model_label(log_path: Path) -> str:
    name = log_path.stem
    for prefix in ("agent_", "run_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    name = re.sub(r"_\d{8}_\d{6}$", "", name)
    replacements = {
        "gemini-3-flash-preview": "Gemini3",
        "qwen3vl8b-vllm": "Qwen3VL",
        "qwen3-vl-8b": "Qwen3VL",
        "qwen38b-vllm": "Qwen3",
        "qwen3-8b": "Qwen3",
        "llama-3.1-8b-vllm": "LLaMA3.1",
        "mistral-7b-vllm": "Ministral-3",
        "ministral-3-8b": "Ministral-3",
    }
    return replacements.get(name, name)


def build_paper_figure2_data(
    utility_gap_df: pd.DataFrame,
    dataset_label: str,
    model_label: str,
) -> pd.DataFrame:
    if utility_gap_df.empty:
        return pd.DataFrame(columns=["dataset", "model", "attribute", "tool", "delta_acc_abs_pct"])
    df = utility_gap_df.copy()
    df["dataset"] = dataset_label
    df["model"] = model_label
    df["tool"] = df["tool"].map(paper_tool_label)
    df["delta_acc_abs_pct"] = pd.to_numeric(df["uplift_gap"], errors="coerce").abs() * 100.0
    df = (
        df.groupby(["dataset", "model", "attribute", "tool"], as_index=False)["delta_acc_abs_pct"]
        .mean()
        .sort_values(["dataset", "attribute", "tool", "model"])
    )
    return df


def plot_paper_figure2(fig2_df: pd.DataFrame, out_path: Path) -> None:
    if fig2_df.empty:
        return
    datasets = [d for d in ["CheXAgentBench", "MIMIC-FairnessVQA"] if d in set(fig2_df["dataset"])]
    attrs = [a for a in ["gender_norm", "age_group"] if a in set(fig2_df["attribute"])]
    if not datasets or not attrs:
        return
    fig, axes = plt.subplots(len(datasets), len(attrs), figsize=(5.6 * len(attrs), 4.2 * len(datasets)), squeeze=False)
    rng = np.random.default_rng(42)
    titles = {"gender_norm": "Gender", "age_group": "Age"}
    for r, dataset in enumerate(datasets):
        for c, attr in enumerate(attrs):
            ax = axes[r][c]
            sub = fig2_df[(fig2_df["dataset"] == dataset) & (fig2_df["attribute"] == attr)]
            positions = np.arange(len(PAPER_TOOL_ORDER))
            values_by_tool = [
                sub[sub["tool"] == tool]["delta_acc_abs_pct"].dropna().to_numpy(dtype=float)
                for tool in PAPER_TOOL_ORDER
            ]
            nonempty = [(pos, vals) for pos, vals in zip(positions, values_by_tool) if len(vals) > 0]
            if nonempty:
                ax.violinplot(
                    [vals for _, vals in nonempty],
                    positions=[pos for pos, _ in nonempty],
                    widths=0.72,
                    showmeans=True,
                    showextrema=False,
                )
                for pos, vals in nonempty:
                    jitter = rng.normal(0, 0.035, size=len(vals))
                    ax.scatter(np.full(len(vals), pos) + jitter, vals, s=24, color="black", alpha=0.75, zorder=3)
            ax.set_xticks(positions)
            ax.set_xticklabels(PAPER_TOOL_ORDER)
            ax.set_ylabel(r"$|\Delta ACC|$ conditioned on tool (%)")
            ax.set_title(f"{titles.get(attr, attr)} on {dataset}")
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_paper_figure3_data(
    transition_df: pd.DataFrame,
    dataset_label: str,
    model_label: str,
) -> pd.DataFrame:
    if transition_df.empty:
        return pd.DataFrame(columns=["dataset", "model", "attribute", "from_tool", "to_tool", "delta_rate"])
    rows: List[Dict[str, Any]] = []
    desired_pairs = {
        "gender_norm": ("male", "female"),
        "age_group": ("young", "old"),
    }
    df = transition_df.copy()
    split = df["transition"].astype(str).str.split("->", n=1, expand=True)
    df["from_tool"] = split[0].map(paper_tool_label)
    df["to_tool"] = split[1].map(paper_tool_label)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = (
        df.groupby(["attribute", "group", "from_tool", "to_tool"], as_index=False)["rate"]
        .sum()
    )
    for attr, (positive_group, negative_group) in desired_pairs.items():
        sub = df[df["attribute"] == attr]
        if sub.empty:
            continue
        piv = sub.pivot_table(
            index=["from_tool", "to_tool"],
            columns="group",
            values="rate",
            aggfunc="sum",
            fill_value=0.0,
        )
        if positive_group not in piv.columns or negative_group not in piv.columns:
            continue
        for (from_tool, to_tool), vals in piv.iterrows():
            if from_tool not in PAPER_TRANSITION_ORDER or to_tool not in PAPER_TRANSITION_ORDER:
                continue
            rows.append(
                {
                    "dataset": dataset_label,
                    "model": model_label,
                    "attribute": attr,
                    "from_tool": from_tool,
                    "to_tool": to_tool,
                    "delta_rate": float(vals[positive_group] - vals[negative_group]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["dataset", "model", "attribute", "from_tool", "to_tool", "delta_rate"])
    return (
        out.groupby(["dataset", "model", "attribute", "from_tool", "to_tool"], as_index=False)["delta_rate"]
        .mean()
        .sort_values(["dataset", "attribute", "from_tool", "to_tool", "model"])
    )


def plot_paper_figure3(fig3_df: pd.DataFrame, out_path: Path) -> None:
    if fig3_df.empty:
        return
    datasets = [d for d in ["CheXAgentBench", "MIMIC-FairnessVQA"] if d in set(fig3_df["dataset"])]
    attrs = [a for a in ["gender_norm", "age_group"] if a in set(fig3_df["attribute"])]
    if not datasets or not attrs:
        return
    fig, axes = plt.subplots(
        len(datasets),
        len(attrs),
        figsize=(5.4 * len(attrs) + 0.4, 4.8 * len(datasets)),
        squeeze=False,
    )
    fig.subplots_adjust(right=0.88, wspace=0.34, hspace=0.38)
    titles = {"gender_norm": r"$P_{male} - P_{female}$", "age_group": r"$P_{young} - P_{old}$"}
    max_abs = float(np.nanmax(np.abs(fig3_df["delta_rate"].to_numpy(dtype=float)))) if len(fig3_df) else 0.0
    vmax = max(max_abs, 1e-6)
    for r, dataset in enumerate(datasets):
        for c, attr in enumerate(attrs):
            ax = axes[r][c]
            sub = fig3_df[(fig3_df["dataset"] == dataset) & (fig3_df["attribute"] == attr)]
            mat = (
                sub.pivot_table(index="from_tool", columns="to_tool", values="delta_rate", aggfunc="mean")
                .reindex(index=PAPER_TRANSITION_ORDER, columns=PAPER_TOOL_ORDER)
                .fillna(0.0)
            )
            im = ax.imshow(mat.to_numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(np.arange(len(PAPER_TOOL_ORDER)))
            ax.set_xticklabels(PAPER_TOOL_ORDER, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(PAPER_TRANSITION_ORDER)))
            ax.set_yticklabels(PAPER_TRANSITION_ORDER)
            ax.set_title(f"{dataset}: {titles.get(attr, attr)}")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat.iloc[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Transition-rate difference", rotation=90)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_paper_figures_only_outputs(
    out_dir: Path,
    log_path: Path,
    meta_q_path: Path,
    dataset_label: Optional[str] = None,
    model_label: Optional[str] = None,
) -> None:
    dataset = dataset_label or infer_dataset_label(meta_q_path, log_path)
    model = model_label or infer_model_label(log_path)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    utility_gap = pd.read_csv(out_dir / "lens_agentic_conditional_tool_utility_gap.csv")
    transition_rates = pd.read_csv(out_dir / "lens_agentic_transition_rates_by_group.csv")
    fig2_df = build_paper_figure2_data(utility_gap, dataset, model)
    fig3_df = build_paper_figure3_data(transition_rates, dataset, model)
    fig2_df.to_csv(out_dir / "paper_figure2_tool_exposure_data.csv", index=False)
    fig3_df.to_csv(out_dir / "paper_figure3_tool_transition_data.csv", index=False)
    plot_paper_figure2(fig2_df, fig_dir / "paper_figure2_tool_exposure_bias.png")
    plot_paper_figure3(fig3_df, fig_dir / "paper_figure3_tool_transition_bias.png")

    manifest = {
        "mode": "paper_figures_only",
        "dataset": dataset,
        "model": model,
        "source_log": str(log_path),
        "figures": sorted(p.name for p in fig_dir.glob("paper_figure*.png")),
        "data": [
            "paper_figure2_tool_exposure_data.csv",
            "paper_figure3_tool_transition_data.csv",
        ],
        "note": "Single-log diagnostic panels only. Use analysis/generate_paper_figures.py with all five driver-LLM fairness_posthoc outputs for both datasets to reproduce the full paper figures.",
    }
    (out_dir / "paper_figure_manifest.json").write_text(json.dumps(manifest, indent=2))

    allowed_root = {
        "paper_figure2_tool_exposure_data.csv",
        "paper_figure3_tool_transition_data.csv",
        "paper_figure_manifest.json",
    }
    for path in out_dir.iterdir():
        if path.is_file() and path.name not in allowed_root:
            path.unlink()
    for path in fig_dir.glob("*"):
        if path.name not in set(manifest["figures"]):
            path.unlink()


def top_abs_gap_by_col(df: pd.DataFrame, group_col: str, value_cols: List[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(columns=["metric", "max", "min", "abs_gap"])
    for c in value_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            rows.append(
                {
                    "metric": c,
                    "max": float(s.max()),
                    "min": float(s.min()),
                    "abs_gap": float(s.max() - s.min()),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["metric", "max", "min", "abs_gap"])
    out = pd.DataFrame(rows).sort_values("abs_gap", ascending=False)
    return out


def call_count_bucket(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0)
    bins = pd.cut(x, bins=[-1, 0, 2, 5, 9999], labels=["0", "1-2", "3-5", "6+"])
    return bins.astype(str)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return float(0.5 * (kl_pm + kl_qm))


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    if len(a) == 1 and len(b) == 1:
        d = float(a[0] - b[0])
        return (d, d)
    ia = RNG.integers(0, len(a), size=(n_boot, len(a)))
    ib = RNG.integers(0, len(b), size=(n_boot, len(b)))
    da = a[ia].mean(axis=1)
    db = b[ib].mean(axis=1)
    diff = da - db
    lo = np.quantile(diff, alpha / 2)
    hi = np.quantile(diff, 1 - alpha / 2)
    return (float(lo), float(hi))


def age_group(age_val: Any) -> str:
    age = safe_int(age_val)
    if age is None:
        return "UNKNOWN"
    if age < 18:
        return "0-17"
    if age < 40:
        return "18-39"
    if age < 60:
        return "40-59"
    if age < 80:
        return "60-79"
    return "80+"


def normalize_age_group_label(label: Any) -> str:
    s = str(label).strip().lower() if label is not None else ""
    if s in {"", "unknown", "nan", "none", "null"}:
        return "UNKNOWN"
    if s == "yongd":
        return "young"
    return s


def age_group_chex_binary(age_val: Any, age_label: Any = None) -> str:
    """CheXAgentBench age binning: binary groups only."""
    age = safe_int(age_val)
    if age is not None:
        return "old" if age >= 60 else "young"

    label = normalize_age_group_label(age_label)
    if label in {"", "unknown", "nan", "none", "null"}:
        return "UNKNOWN"
    if label in {"60-79", "80+", "old"}:
        return "old"
    if label in {"0-17", "18-39", "40-59", "young"}:
        return "young"
    return "UNKNOWN"


def use_chex_binary_age_groups(meta_q_path: Path, log_path: Path) -> bool:
    s = f"{str(meta_q_path).lower()} {str(log_path).lower()}"
    return ("chexagentbench" in s) or ("chestagentbench" in s)


def pick_baseline_log(baseline_dir: Path, baseline_name: str) -> Path | None:
    """
    Pick a primary log file for a baseline directory.
    Preference order:
      1) files named like {baseline_name}_*.json or .jsonl (excluding tool_calls_*)
      2) any .json or .jsonl (excluding tool_calls_*)
    If multiple matches, choose the most recently modified.
    """
    preferred: List[str] = []
    preferred.extend(glob(str(baseline_dir / f"{baseline_name}_*.json")))
    preferred.extend(glob(str(baseline_dir / f"{baseline_name}_*.jsonl")))
    preferred = [p for p in preferred if "tool_calls_" not in Path(p).name]

    candidates = preferred
    if not candidates:
        fallback: List[str] = []
        fallback.extend(glob(str(baseline_dir / "*.json")))
        fallback.extend(glob(str(baseline_dir / "*.jsonl")))
        candidates = [p for p in fallback if "tool_calls_" not in Path(p).name]

    if not candidates:
        return None

    return Path(max(candidates, key=lambda p: Path(p).stat().st_mtime))


def run_single(
    log_path: Path,
    out_dir: Path,
    meta_q_path: Path,
    meta_case_path: Path,
    keep_think_text: bool = False,
    enable_llm_judge: bool = False,
    judge_model: str = "deepseek-chat",
    judge_base_url: str = "https://api.deepseek.com/v1",
    judge_api_key_env: str = "DEEPSEEK_API_KEY",
    judge_max_samples: Optional[int] = None,
    judge_concurrency: int = 10,
    paper_figures_only: bool = False,
    paper_dataset_label: Optional[str] = None,
    paper_model_label: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    meta_q = read_metadata_question(meta_q_path) # all original questions
    q_to_case = dict(zip(meta_q["question_id"], meta_q["case_id"].astype(str)))
    q_type = dict(zip(meta_q["question_id"], meta_q["type"].astype(str)))
    q_text = dict(zip(meta_q["question_id"], meta_q["question"].astype(str)))
    q_sections = dict(zip(meta_q["question_id"], meta_q["sections"].astype(str)))
    q_images = dict(zip(meta_q["question_id"], meta_q["images"]))
    chex_binary_age_mode = use_chex_binary_age_groups(meta_q_path, log_path)

    case_meta = read_metadata_case(meta_case_path) # PosixPath('./data/eurorad_metadata.json')

    rows = []
    tool_names = Counter()
    tool_fail_names = Counter()
    tool_first = Counter()
    seq_counts = Counter()

    with log_path.open() as f:
        for line in f:
            line = line.strip()
            entry = json.loads(line)
            qid = entry.get("question_id")
            cid = q_to_case.get(qid)
            question_text = q_text.get(qid)
            question_sections = [
                s.strip() for s in parse_list_like(q_sections.get(qid, ""))
                if s and s.strip() and s.strip().lower() not in {"nan", "none", "null"}
            ]
            question_images = parse_list_like(q_images.get(qid))
            trace = entry.get("trace") or []
            tool_calls, tool_outputs, ai_texts = parse_trace(trace)

            tool_seq = [tc.name for tc in tool_calls if tc.name]

            if tool_seq:
                tool_first[tool_seq[0]] += 1
                seq_counts["->".join(tool_seq)] += 1

            # tool output failures
            tool_fail_flags = []
            for name, output in tool_outputs:
                if name:
                    tool_names[name] += 1
                failed = tool_output_failed(output)
                tool_fail_flags.append((name, failed))
                if name and failed:
                    tool_fail_names[name] += 1
            # ensure tools referenced in calls are tracked even if no output logged
            for name in tool_seq:
                if name:
                    tool_names[name] += 0

            final_text = ""
            if ai_texts:
                final_text = ai_texts[-1]
            final_text_no_think = strip_think(final_text)
            text_for_features = final_text if keep_think_text else final_text_no_think

            # text features
            hedges = count_patterns(text_for_features, HEDGE_PATTERNS)
            certs = count_patterns(text_for_features, CERT_PATTERNS)
            refusals = count_patterns(text_for_features, REFUSAL_PATTERNS)
            demos = count_patterns(text_for_features, DEMOGRAPHIC_PATTERNS)
            inconsist = count_patterns(text_for_features, INCONSISTENCY_PATTERNS)

            # refusal flags via additional kwargs if present
            refusal_flag = False
            for t in trace:
                add = t.get("additional_kwargs")
                if isinstance(add, dict) and add.get("refusal"):
                    refusal_flag = True
                    break
            if refusals > 0:
                refusal_flag = True

            # attributes
            case_attrs = case_meta.get(str(cid), {}) if cid is not None else {}
            gender = case_attrs.get("gender", "UNKNOWN") if case_attrs else "UNKNOWN"
            if gender in (None, "", "nan"):
                gender = "UNKNOWN"
            gender_norm = str(gender).strip().lower().replace(",", "")
            if gender_norm in ("man", "male", "m"):
                gender_norm = "male"
            elif gender_norm in ("woman", "female", "f"):
                gender_norm = "female"
            elif gender_norm in ("unknown", "", "nan", "none"):
                gender_norm = "UNKNOWN"
            age_raw = case_attrs.get("age", case_attrs.get("anchor_age", "UNKNOWN")) if case_attrs else "UNKNOWN"
            age_label = case_attrs.get("age_group", "UNKNOWN") if case_attrs else "UNKNOWN"
            if chex_binary_age_mode:
                age_grp = age_group_chex_binary(age_raw, age_label)
            elif age_label in (None, "", "nan", "none", "null"):
                age_grp = age_group(age_raw)
            else:
                age_grp = normalize_age_group_label(age_label)
            intersection_group = f"{gender_norm}|{age_grp}"

            # prompt sensitivity proxies (observational)
            prompt_demo_count = count_patterns(question_text, PROMPT_DEMOGRAPHIC_PATTERNS)
            prompt_demographic_explicit = 1 if prompt_demo_count > 0 else 0
            prompt_specificity_bucket = "demographic_explicit" if prompt_demographic_explicit else "neutral"

            # missing context features
            core_case_fields = [
                "history",
                "image_finding",
                "discussion",
                "differential_diagnosis",
                "diagnosis",
                "imaging_technique",
                "figures",
            ]
            case_present = sum(1 for k in core_case_fields if has_value(case_attrs.get(k)))
            case_total = len(core_case_fields)
            case_context_completeness = float(case_present / case_total) if case_total else np.nan

            requested_sections_present = 0
            for sec in question_sections:
                requested_sections_present += 1 if has_value(case_attrs.get(sec)) else 0
            question_sections_count = len(question_sections)
            requested_context_completeness = (
                float(requested_sections_present / question_sections_count) if question_sections_count else np.nan
            )

            image_count = len(question_images)
            image_missing_flag = 1 if image_count == 0 else 0
            missing_context_flag = 1 if (
                (not np.isnan(requested_context_completeness) and requested_context_completeness < 1.0)
                or image_missing_flag == 1
            ) else 0

            # tool usage indicators
            tool_used_set = set(tool_seq)
            tool_used_flags = {f"tool_used__{t}": (1 if t in tool_used_set else 0) for t in tool_names.keys()}

            # tool failure indicators
            tool_fail_flags_dict = defaultdict(int)
            for name, failed in tool_fail_flags:
                if not name:
                    continue
                if failed:
                    tool_fail_flags_dict[f"tool_failed__{name}"] += 1
            # per-tool failure count per question

            rows.append(
                {
                    "question_id": qid,
                    "case_id": cid,
                    "question_type": q_type.get(qid, "UNKNOWN"),
                    "question_text": question_text,
                    "status": entry.get("status"),
                    "timestamp": entry.get("timestamp"),
                    "model": entry.get("model"),
                    "temperature": entry.get("temperature"),
                    "attempts": entry.get("attempts"),
                    "is_correct": entry.get("is_correct"),
                    "predicted_answer": entry.get("predicted_answer"),
                    "correct_answer": entry.get("correct_answer"),
                    "model_answer": entry.get("model_answer"),
                    "tool_call_count": len(tool_seq),
                    "unique_tool_count": len(set(tool_seq)),
                    "first_tool": tool_seq[0] if tool_seq else "NONE",
                    "tool_sequence": "->".join(tool_seq) if tool_seq else "NONE",
                    "tool_failure_any": 1 if any(f for _, f in tool_fail_flags) else 0,
                    "final_response": final_text,
                    "final_response_no_think": final_text_no_think,
                    "response_chars": len(text_for_features),
                    "response_words": len(text_for_features.split()),
                    "hedge_count": hedges,
                    "certainty_count": certs,
                    "refusal_count": refusals,
                    "refusal_flag": 1 if refusal_flag else 0,
                    "demographic_terms": demos,
                    "inconsistency_markers": inconsist,
                    "feature_text_mode": "with_think" if keep_think_text else "no_think",
                    "gender": gender,
                    "gender_norm": gender_norm,
                    "age_raw": age_raw,
                    "age_group": age_grp,
                    "intersection_group": intersection_group,
                    "prompt_demographic_explicit": prompt_demographic_explicit,
                    "prompt_demographic_count": prompt_demo_count,
                    "prompt_specificity_bucket": prompt_specificity_bucket,
                    "question_sections_count": question_sections_count,
                    "requested_sections_present": requested_sections_present,
                    "requested_context_completeness": requested_context_completeness,
                    "case_context_completeness": case_context_completeness,
                    "question_image_count": image_count,
                    "image_missing_flag": image_missing_flag,
                    "missing_context_flag": missing_context_flag,
                    **tool_used_flags,
                    **tool_fail_flags_dict,
                }
            )

    df = pd.DataFrame(rows)

    # ensure tool_used columns exist for all tools
    for t in sorted(tool_names.keys()):
        col = f"tool_used__{t}"
        if col not in df.columns:
            df[col] = 0
        fail_col = f"tool_failed__{t}"
        if fail_col not in df.columns:
            df[fail_col] = 0

    # write per-question dataset
    per_q_path = out_dir / "per_question_features.csv"
    df.to_csv(per_q_path, index=False)

    if enable_llm_judge:
        df = score_reasoning_with_deepseek(
            df=df,
            out_dir=out_dir,
            model=judge_model,
            base_url=judge_base_url,
            api_key_env=judge_api_key_env,
            max_samples=judge_max_samples,
            concurrency=judge_concurrency,
        )
        df.to_csv(per_q_path, index=False)

    # numeric is_correct for analysis
    df["is_correct_num"] = pd.to_numeric(df["is_correct"], errors="coerce")

    # coverage summary
    coverage = {
        "total_questions": len(df),
        "matched_case_id": int(df["case_id"].notna().sum()),
        "matched_gender": int((df["gender_norm"] != "UNKNOWN").sum()),
        "matched_age": int((df["age_group"] != "UNKNOWN").sum()),
        "prompt_demographic_explicit_n": int((df["prompt_demographic_explicit"] == 1).sum()),
        "missing_context_n": int((df["missing_context_flag"] == 1).sum()),
        "status_ok": int((df["status"] == "ok").sum()),
        "status_not_ok": int((df["status"] != "ok").sum()),
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2))

    # identify sensitive attributes to analyze
    sensitive_attrs = ["gender_norm", "age_group"]

    # group-wise performance
    perf_rows = []
    for attr in sensitive_attrs:
        for grp, sub in df.groupby(attr, dropna=False):
            values = sub["is_correct_num"].to_numpy()
            acc = float(np.nanmean(values)) if len(values) else np.nan
            lo, hi, bstd = bootstrap_summary(values, n_boot=BOOTSTRAP_SAMPLES)
            perf_rows.append(
                {
                    "attribute": attr,
                    "group": grp,
                    "n": len(sub),
                    "accuracy": acc,
                    "bootstrap_std": bstd,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap_n": BOOTSTRAP_SAMPLES,
                }
            )
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(out_dir / "group_performance.csv", index=False)

    # intersectional group-wise performance
    inter_rows = []
    for grp, sub in df.groupby("intersection_group", dropna=False):
        values = sub["is_correct_num"].to_numpy()
        acc = float(np.nanmean(values)) if len(values) else np.nan
        lo, hi, bstd = bootstrap_summary(values, n_boot=BOOTSTRAP_SAMPLES)
        inter_rows.append(
            {
                "attribute": "gender_norm_x_age_group",
                "group": grp,
                "n": len(sub),
                "accuracy": acc,
                "bootstrap_std": bstd,
                "ci_low": lo,
                "ci_high": hi,
                "bootstrap_n": BOOTSTRAP_SAMPLES,
            }
        )
    inter_df = pd.DataFrame(inter_rows).sort_values("accuracy", ascending=False)
    inter_df.to_csv(out_dir / "group_performance_intersectional.csv", index=False)

    # performance by question type
    perf_qtype_rows = []
    for attr in sensitive_attrs:
        for (grp, qtype), sub in df.groupby([attr, "question_type"], dropna=False):
            values = sub["is_correct_num"].to_numpy()
            acc = float(np.nanmean(values)) if len(values) else np.nan
            lo, hi, bstd = bootstrap_summary(values, n_boot=BOOTSTRAP_SAMPLES)
            perf_qtype_rows.append(
                {
                    "attribute": attr,
                    "group": grp,
                    "question_type": qtype,
                    "n": len(sub),
                    "accuracy": acc,
                    "bootstrap_std": bstd,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap_n": BOOTSTRAP_SAMPLES,
                }
            )
    pd.DataFrame(perf_qtype_rows).to_csv(out_dir / "group_performance_by_question_type.csv", index=False)

    # tool usage disparities
    tool_usage_rows = []
    tool_list = sorted(tool_names.keys())
    for attr in sensitive_attrs:
        for grp, sub in df.groupby(attr, dropna=False):
            row = {
                "attribute": attr,
                "group": grp,
                "n": len(sub),
                "tool_call_rate": float(sub["tool_call_count"].mean()),
            }
            for t in tool_list:
                row[f"tool_used_rate__{t}"] = float(sub[f"tool_used__{t}"].mean())
                row[f"tool_fail_rate__{t}"] = float(sub[f"tool_failed__{t}"].mean())
            tool_usage_rows.append(row)
    tool_usage_df = pd.DataFrame(
        tool_usage_rows,
        columns=["attribute", "group", "n", "tool_call_rate"] + [f"tool_used_rate__{t}" for t in tool_list] + [f"tool_fail_rate__{t}" for t in tool_list],
    )
    tool_usage_df.to_csv(out_dir / "tool_usage_by_group.csv", index=False)

    # first tool distribution
    first_tool_rows = []
    for attr in sensitive_attrs:
        for (grp, first), sub in df.groupby([attr, "first_tool"], dropna=False):
            first_tool_rows.append(
                {
                    "attribute": attr,
                    "group": grp,
                    "first_tool": first,
                    "n": len(sub),
                }
            )
    pd.DataFrame(first_tool_rows).to_csv(out_dir / "first_tool_by_group.csv", index=False)

    # common sequences per group (top 10)
    seq_rows = []
    for attr in sensitive_attrs:
        for grp, sub in df.groupby(attr, dropna=False):
            seq_counts_grp = Counter(sub["tool_sequence"]).most_common(10)
            for seq, n in seq_counts_grp:
                seq_rows.append(
                    {
                        "attribute": attr,
                        "group": grp,
                        "tool_sequence": seq,
                        "n": n,
                    }
                )
    pd.DataFrame(seq_rows).to_csv(out_dir / "tool_sequences_top10_by_group.csv", index=False)

    # outcome conditional on tool usage
    strat_rows = []
    for attr in sensitive_attrs:
        for grp, sub in df.groupby(attr, dropna=False):
            for t in tool_list:
                used = sub[sub[f"tool_used__{t}"] == 1]
                not_used = sub[sub[f"tool_used__{t}"] == 0]
                used_acc = float(np.nanmean(used["is_correct_num"])) if len(used) else np.nan
                not_acc = float(np.nanmean(not_used["is_correct_num"])) if len(not_used) else np.nan
                strat_rows.append(
                    {
                        "attribute": attr,
                        "group": grp,
                        "tool": t,
                        "n_used": len(used),
                        "n_not_used": len(not_used),
                        "acc_used": used_acc,
                        "acc_not_used": not_acc,
                        "acc_diff_used_minus_not": (used_acc - not_acc) if (len(used) and len(not_used)) else np.nan,
                    }
                )
    strat_df = pd.DataFrame(
        strat_rows,
        columns=[
            "attribute",
            "group",
            "tool",
            "n_used",
            "n_not_used",
            "acc_used",
            "acc_not_used",
            "acc_diff_used_minus_not",
        ],
    )
    strat_df.to_csv(out_dir / "outcome_by_tool_usage_within_group.csv", index=False)

    # prompt sensitivity analysis (observational)
    prompt_rows = []
    for bucket, sub in df.groupby("prompt_specificity_bucket", dropna=False):
        values = sub["is_correct_num"].to_numpy()
        acc = float(np.nanmean(values)) if len(values) else np.nan
        lo, hi = bootstrap_ci(values)
        prompt_rows.append(
            {
                "prompt_bucket": bucket,
                "n": len(sub),
                "accuracy": acc,
                "ci_low": lo,
                "ci_high": hi,
                "tool_call_rate": float(sub["tool_call_count"].mean()),
                "refusal_rate": float(sub["refusal_flag"].mean()),
                "hedge_rate": float(sub["hedge_count"].mean()),
            }
        )
    prompt_df = pd.DataFrame(prompt_rows).sort_values("prompt_bucket")
    prompt_df.to_csv(out_dir / "prompt_sensitivity_summary.csv", index=False)

    prompt_group_rows = []
    for attr in sensitive_attrs:
        for (grp, bucket), sub in df.groupby([attr, "prompt_specificity_bucket"], dropna=False):
            values = sub["is_correct_num"].to_numpy()
            acc = float(np.nanmean(values)) if len(values) else np.nan
            prompt_group_rows.append(
                {
                    "attribute": attr,
                    "group": grp,
                    "prompt_bucket": bucket,
                    "n": len(sub),
                    "accuracy": acc,
                    "tool_call_rate": float(sub["tool_call_count"].mean()),
                }
            )
    prompt_group_df = pd.DataFrame(prompt_group_rows)
    prompt_group_df.to_csv(out_dir / "prompt_sensitivity_by_group.csv", index=False)

    # missing-context analysis
    def completeness_bucket(v: float) -> str:
        if np.isnan(v):
            return "unknown"
        if v >= 0.99:
            return "full"
        if v >= 0.66:
            return "medium"
        return "low"

    df["requested_context_bucket"] = df["requested_context_completeness"].apply(completeness_bucket)

    miss_rows = []
    for bucket, sub in df.groupby("requested_context_bucket", dropna=False):
        values = sub["is_correct_num"].to_numpy()
        acc = float(np.nanmean(values)) if len(values) else np.nan
        lo, hi = bootstrap_ci(values)
        miss_rows.append(
            {
                "context_bucket": bucket,
                "n": len(sub),
                "accuracy": acc,
                "ci_low": lo,
                "ci_high": hi,
                "tool_call_rate": float(sub["tool_call_count"].mean()),
                "failure_any_rate": float(sub["tool_failure_any"].mean()),
            }
        )
    miss_df = pd.DataFrame(miss_rows).sort_values("context_bucket")
    miss_df.to_csv(out_dir / "missing_context_summary.csv", index=False)

    miss_group_rows = []
    for attr in sensitive_attrs:
        for (grp, bucket), sub in df.groupby([attr, "requested_context_bucket"], dropna=False):
            values = sub["is_correct_num"].to_numpy()
            acc = float(np.nanmean(values)) if len(values) else np.nan
            miss_group_rows.append(
                {
                    "attribute": attr,
                    "group": grp,
                    "context_bucket": bucket,
                    "n": len(sub),
                    "accuracy": acc,
                    "tool_call_rate": float(sub["tool_call_count"].mean()),
                }
            )
    miss_group_df = pd.DataFrame(miss_group_rows)
    miss_group_df.to_csv(out_dir / "missing_context_by_group.csv", index=False)

    # LLM output characteristics by group
    text_rows = []
    text_cols = [
        "response_chars",
        "response_words",
        "hedge_count",
        "certainty_count",
        "refusal_count",
        "refusal_flag",
        "demographic_terms",
        "inconsistency_markers",
    ]
    if "reasoning_quality_score" in df.columns:
        text_cols.append("reasoning_quality_score")
    for attr in sensitive_attrs:
        for grp, sub in df.groupby(attr, dropna=False):
            row = {"attribute": attr, "group": grp, "n": len(sub)}
            for c in text_cols:
                row[c] = float(pd.to_numeric(sub[c], errors="coerce").mean())
            text_rows.append(row)
    text_df = pd.DataFrame(
        text_rows,
        columns=["attribute", "group", "n"] + text_cols,
    )
    text_df.to_csv(out_dir / "llm_text_features_by_group.csv", index=False)

    # association tests (chi-square where possible)
    assoc_rows = []
    try:
        from scipy.stats import chi2_contingency  # type: ignore

        tested_attrs = sensitive_attrs + ["intersection_group", "prompt_specificity_bucket", "requested_context_bucket"]
        for attr in tested_attrs:
            # outcome vs group
            cont = pd.crosstab(df[attr], df["is_correct"])
            if cont.shape[0] > 1 and cont.shape[1] > 1:
                chi2, p, _, _ = chi2_contingency(cont)
                assoc_rows.append({"attribute": attr, "test": "group_vs_outcome", "chi2": chi2, "p": p})

            for t in tool_list:
                cont = pd.crosstab(df[attr], df[f"tool_used__{t}"])
                if cont.shape[0] > 1 and cont.shape[1] > 1:
                    chi2, p, _, _ = chi2_contingency(cont)
                    assoc_rows.append(
                        {
                            "attribute": attr,
                            "test": f"group_vs_tool_used__{t}",
                            "chi2": chi2,
                            "p": p,
                        }
                    )
                cont = pd.crosstab(df[attr], df[f"tool_failed__{t}"])
                if cont.shape[0] > 1 and cont.shape[1] > 1:
                    chi2, p, _, _ = chi2_contingency(cont)
                    assoc_rows.append(
                        {
                            "attribute": attr,
                            "test": f"group_vs_tool_failed__{t}",
                            "chi2": chi2,
                            "p": p,
                        }
                    )
    except Exception as e:
        assoc_rows.append({"attribute": "ALL", "test": "chi_square", "chi2": np.nan, "p": np.nan, "error": str(e)})

    assoc_df = pd.DataFrame(assoc_rows)
    if not assoc_df.empty and "p" in assoc_df.columns:
        valid_mask = assoc_df["p"].notna()
        p_vals = assoc_df.loc[valid_mask, "p"].astype(float).tolist()
        fdr_vals = bh_fdr(p_vals)
        assoc_df["p_fdr_bh"] = np.nan
        assoc_df.loc[valid_mask, "p_fdr_bh"] = fdr_vals
    assoc_df.to_csv(out_dir / "association_tests.csv", index=False)

    # attribution-style summary (heuristic)
    summary_lines = []
    summary_lines.append("# Post-hoc Fairness Analysis (Logs Only)")
    summary_lines.append("")
    summary_lines.append("## Data Integrity / Coverage")
    summary_lines.append(f"- Total questions: {coverage['total_questions']}")
    summary_lines.append(f"- Matched case_id: {coverage['matched_case_id']}")
    summary_lines.append(f"- Matched gender: {coverage['matched_gender']}")
    summary_lines.append(f"- Matched age_group: {coverage['matched_age']}")
    summary_lines.append(f"- Prompt demographic explicit count: {coverage['prompt_demographic_explicit_n']}")
    summary_lines.append(f"- Missing context flagged count: {coverage['missing_context_n']}")
    summary_lines.append(f"- Status ok: {coverage['status_ok']} | not ok: {coverage['status_not_ok']}")
    summary_lines.append("")

    attr_label = {"gender_norm": "gender", "age_group": "age_group"}

    summary_lines.append("## Group-wise Performance Disparities")
    if not perf_df.empty:
        for attr in sensitive_attrs:
            sub = perf_df[perf_df["attribute"] == attr].sort_values("accuracy")
            summary_lines.append(f"- {attr_label.get(attr, attr)}:")
            for _, r in sub.iterrows():
                summary_lines.append(
                    f"  - {r['group']}: acc={r['accuracy']:.3f} "
                    f"(n={int(r['n'])}, boot_std={r['bootstrap_std']:.3f}, CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}], "
                    f"boot_n={int(r['bootstrap_n'])})"
                )
    summary_lines.append("")

    # heuristic attribution
    summary_lines.append("## Attribution-Style Summary (Heuristic)")
    for attr in sensitive_attrs:
        sub = perf_df[perf_df["attribute"] == attr].sort_values("accuracy")
        if len(sub) < 2:
            summary_lines.append(f"- {attr_label.get(attr, attr)}: insufficient group count for disparity assessment.")
            continue
        worst = sub.iloc[0]
        best = sub.iloc[-1]
        gap = best["accuracy"] - worst["accuracy"]
        summary_lines.append(f"- {attr_label.get(attr, attr)}: accuracy gap {gap:.3f} between {worst['group']} and {best['group']}.")

        # check tool failure disparity
        tool_usage = pd.read_csv(out_dir / "tool_usage_by_group.csv")
        tu = tool_usage[tool_usage["attribute"] == attr]
        failure_cols = [c for c in tu.columns if c.startswith("tool_fail_rate__")]
        if failure_cols:
            max_fail_tool = None
            max_fail_gap = 0.0
            for c in failure_cols:
                vals = tu.set_index("group")[c]
                if vals.max() - vals.min() > max_fail_gap:
                    max_fail_gap = vals.max() - vals.min()
                    max_fail_tool = c.replace("tool_fail_rate__", "")
            if max_fail_tool and max_fail_gap > 0.05:
                summary_lines.append(
                    f"  - Tool failure disparity suggests inherited tool bias risk (largest failure rate gap in {max_fail_tool}: {max_fail_gap:.3f})."
                )

        # check tool usage disparity
        usage_cols = [c for c in tu.columns if c.startswith("tool_used_rate__")]
        max_use_tool = None
        max_use_gap = 0.0
        for c in usage_cols:
            vals = tu.set_index("group")[c]
            if vals.max() - vals.min() > max_use_gap:
                max_use_gap = vals.max() - vals.min()
                max_use_tool = c.replace("tool_used_rate__", "")
        if max_use_tool and max_use_gap > 0.10:
            summary_lines.append(
                f"  - Tool selection differs by group (largest usage rate gap in {max_use_tool}: {max_use_gap:.3f}), consistent with agentic bias risk."
            )

        # LLM text disparity
        text_df = pd.read_csv(out_dir / "llm_text_features_by_group.csv")
        td = text_df[text_df["attribute"] == attr]
        if len(td) >= 2:
            diff = td.set_index("group")["hedge_count"].max() - td.set_index("group")["hedge_count"].min()
            if diff > 0.5:
                summary_lines.append(
                    f"  - LLM hedging differs across groups (hedge_count gap {diff:.2f}), consistent with reasoning bias risk."
                )

    summary_lines.append("")
    summary_lines.append("## Prompt Sensitivity (Observational Proxy)")
    if not prompt_df.empty:
        for _, r in prompt_df.iterrows():
            summary_lines.append(
                f"- {r['prompt_bucket']}: acc={r['accuracy']:.3f} (n={int(r['n'])}, CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]), "
                f"tool_call_rate={r['tool_call_rate']:.2f}, refusal_rate={r['refusal_rate']:.2f}"
            )

    summary_lines.append("")
    summary_lines.append("## Missing Context Robustness")
    if not miss_df.empty:
        for _, r in miss_df.iterrows():
            summary_lines.append(
                f"- {r['context_bucket']}: acc={r['accuracy']:.3f} (n={int(r['n'])}, CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]), "
                f"failure_any_rate={r['failure_any_rate']:.2f}"
            )

    (out_dir / "summary.md").write_text("\n".join(summary_lines))

    # Visualization outputs
    perf_plot = perf_df.copy()
    perf_plot["group_label"] = perf_plot["attribute"] + ":" + perf_plot["group"].astype(str)
    save_bar_plot(
        perf_plot.sort_values("accuracy", ascending=False),
        x="group_label",
        y="accuracy",
        title="Accuracy by Sensitive Group",
        out_path=fig_dir / "accuracy_by_group.png",
        rotation=35,
    )

    # Tool usage disparity matrix by group (gender + age only)
    tu = pd.read_csv(out_dir / "tool_usage_by_group.csv")
    for attr in sensitive_attrs:
        tu_attr = tu[tu["attribute"] == attr]
        usage_cols = [c for c in tu_attr.columns if c.startswith("tool_used_rate__")]
        if usage_cols:
            m = tu_attr.set_index("group")[usage_cols]
            m.columns = [c.replace("tool_used_rate__", "") for c in m.columns]
            save_matrix_plot(m, f"Tool Usage Rate by {attr}", fig_dir / f"tool_usage_matrix_{attr}.png")

    first_tool_df = pd.read_csv(out_dir / "first_tool_by_group.csv")
    if not first_tool_df.empty:
        ft = first_tool_df.copy()
        ft["group_label"] = ft["attribute"] + ":" + ft["group"].astype(str)
        top_first = ft.sort_values("n", ascending=False).head(25)
        save_bar_plot(
            top_first,
            x="first_tool",
            y="n",
            title="Top First Tools (All Groups)",
            out_path=fig_dir / "first_tool_distribution.png",
            rotation=30,
        )

    save_bar_plot(
        prompt_df.sort_values("prompt_bucket"),
        x="prompt_bucket",
        y="accuracy",
        title="Prompt Sensitivity: Accuracy by Prompt Bucket",
        out_path=fig_dir / "prompt_sensitivity_accuracy.png",
    )

    save_bar_plot(
        miss_df.sort_values("context_bucket"),
        x="context_bucket",
        y="accuracy",
        title="Missing Context: Accuracy by Completeness Bucket",
        out_path=fig_dir / "missing_context_accuracy.png",
    )

    if not inter_df.empty:
        top_inter = inter_df.sort_values("accuracy", ascending=False).head(25)
        save_bar_plot(
            top_inter,
            x="group",
            y="accuracy",
            title="Intersectional Accuracy (gender x age_group)",
            out_path=fig_dir / "intersectional_accuracy.png",
            rotation=35,
        )

    # Lens 1: Inherited bias from tools
    inherited_rows = []
    for attr in sensitive_attrs:
        tu_attr = tool_usage_df[tool_usage_df["attribute"] == attr]
        st_attr = strat_df[strat_df["attribute"] == attr]
        for t in tool_list:
            fail_col = f"tool_fail_rate__{t}"
            use_col = f"tool_used_rate__{t}"
            fail_gap = np.nan
            use_gap = np.nan
            if fail_col in tu_attr.columns and not tu_attr.empty:
                fail_gap = float(pd.to_numeric(tu_attr[fail_col], errors="coerce").max() - pd.to_numeric(tu_attr[fail_col], errors="coerce").min())
            if use_col in tu_attr.columns and not tu_attr.empty:
                use_gap = float(pd.to_numeric(tu_attr[use_col], errors="coerce").max() - pd.to_numeric(tu_attr[use_col], errors="coerce").min())
            st_t = st_attr[st_attr["tool"] == t]
            harm_est = np.nan
            best_group_acc_used = ""
            best_group_acc_used_value = np.nan
            worst_group_acc_used = ""
            worst_group_acc_used_value = np.nan
            acc_used_gap = np.nan
            acc_used_female = np.nan
            acc_used_male = np.nan
            acc_used_young = np.nan
            acc_used_old = np.nan
            if not st_t.empty:
                harm_est = float(pd.to_numeric(st_t["acc_diff_used_minus_not"], errors="coerce").min())
                st_t_num = st_t.copy()
                st_t_num["acc_used_num"] = pd.to_numeric(st_t_num["acc_used"], errors="coerce")
                st_t_num = st_t_num[st_t_num["acc_used_num"].notna()].copy()
                if not st_t_num.empty:
                    best_ix = st_t_num["acc_used_num"].idxmax()
                    worst_ix = st_t_num["acc_used_num"].idxmin()
                    best_group_acc_used = str(st_t_num.loc[best_ix, "group"])
                    best_group_acc_used_value = float(st_t_num.loc[best_ix, "acc_used_num"])
                    worst_group_acc_used = str(st_t_num.loc[worst_ix, "group"])
                    worst_group_acc_used_value = float(st_t_num.loc[worst_ix, "acc_used_num"])
                    acc_used_gap = best_group_acc_used_value - worst_group_acc_used_value

                    if attr == "gender_norm":
                        f = st_t_num[st_t_num["group"].astype(str).str.lower() == "female"]
                        m = st_t_num[st_t_num["group"].astype(str).str.lower() == "male"]
                        if not f.empty:
                            acc_used_female = float(f["acc_used_num"].iloc[0])
                        if not m.empty:
                            acc_used_male = float(m["acc_used_num"].iloc[0])
                    if attr == "age_group":
                        y = st_t_num[st_t_num["group"].astype(str).str.lower() == "young"]
                        o = st_t_num[st_t_num["group"].astype(str).str.lower() == "old"]
                        if not y.empty:
                            acc_used_young = float(y["acc_used_num"].iloc[0])
                        if not o.empty:
                            acc_used_old = float(o["acc_used_num"].iloc[0])
            inherited_rows.append(
                {
                    "attribute": attr,
                    "tool": t,
                    "tool_failure_gap": fail_gap,
                    "tool_usage_gap": use_gap,
                    "min_acc_diff_used_minus_not": harm_est,
                    "best_group_acc_used": best_group_acc_used,
                    "best_group_acc_used_value": best_group_acc_used_value,
                    "worst_group_acc_used": worst_group_acc_used,
                    "worst_group_acc_used_value": worst_group_acc_used_value,
                    "acc_used_gap": acc_used_gap,
                    "acc_used_female": acc_used_female,
                    "acc_used_male": acc_used_male,
                    "acc_used_young": acc_used_young,
                    "acc_used_old": acc_used_old,
                }
            )
    inherited_df = pd.DataFrame(
        inherited_rows,
        columns=[
            "attribute",
            "tool",
            "tool_failure_gap",
            "tool_usage_gap",
            "min_acc_diff_used_minus_not",
            "best_group_acc_used",
            "best_group_acc_used_value",
            "worst_group_acc_used",
            "worst_group_acc_used_value",
            "acc_used_gap",
            "acc_used_female",
            "acc_used_male",
            "acc_used_young",
            "acc_used_old",
        ],
    )
    inherited_df.to_csv(out_dir / "lens_inherited_tool_bias.csv", index=False)

    for attr in sensitive_attrs:
        sub = inherited_df[inherited_df["attribute"] == attr].dropna(subset=["tool_failure_gap"]) if not inherited_df.empty else pd.DataFrame()
        if not sub.empty:
            top = sub.sort_values("tool_failure_gap", ascending=False).head(15)
            save_bar_plot(
                top,
                x="tool",
                y="tool_failure_gap",
                title=f"Inherited Tool Bias Proxy: Failure Gap by Tool ({attr})",
                out_path=fig_dir / f"lens_inherited_tool_bias_{attr}.png",
                rotation=30,
            )

    # Lens 2: Agentic bias via tool selection
    agentic_rows = []
    for attr in sensitive_attrs:
        tu_attr = tool_usage_df[tool_usage_df["attribute"] == attr]
        usage_cols = [c for c in tu_attr.columns if c.startswith("tool_used_rate__")]
        gap_df = top_abs_gap_by_col(tu_attr, "group", usage_cols)
        for _, r in gap_df.iterrows():
            agentic_rows.append(
                {
                    "attribute": attr,
                    "tool": str(r["metric"]).replace("tool_used_rate__", ""),
                    "usage_gap": float(r["abs_gap"]),
                }
            )
    agentic_df = pd.DataFrame(agentic_rows, columns=["attribute", "tool", "usage_gap"])
    if not agentic_df.empty:
        agentic_df = agentic_df.sort_values(["attribute", "usage_gap"], ascending=[True, False])
    agentic_df.to_csv(out_dir / "lens_agentic_tool_selection_bias.csv", index=False)

    for attr in sensitive_attrs:
        sub = agentic_df[agentic_df["attribute"] == attr].head(15) if not agentic_df.empty else pd.DataFrame()
        if not sub.empty:
            save_bar_plot(
                sub,
                x="tool",
                y="usage_gap",
                title=f"Agentic Selection Bias Proxy: Tool Usage Gap ({attr})",
                out_path=fig_dir / f"lens_agentic_selection_bias_{attr}.png",
                rotation=30,
            )

    # Lens 2a: Path-conditioned subgroup gap (controls for coarse tool trajectory)
    df["plan_bucket"] = df["first_tool"].astype(str).fillna("NONE") + "|calls=" + call_count_bucket(df["tool_call_count"])
    path_rows = []
    path_std_rows = []
    path_summary_rows = []
    for attr in sensitive_attrs:
        attr_df = df[df[attr].notna()].copy()
        if attr_df.empty:
            continue
        attr_df[attr] = attr_df[attr].astype(str)
        attr_df = attr_df[attr_df[attr] != "UNKNOWN"]
        if attr_df.empty:
            continue

        bucket_weights = attr_df["plan_bucket"].value_counts(normalize=True, dropna=False).to_dict()
        group_means = attr_df.groupby(attr)["is_correct_num"].mean().to_dict()
        std_acc = {g: 0.0 for g in group_means.keys()}

        for bucket, bdf in attr_df.groupby("plan_bucket", dropna=False):
            group_perf = (
                bdf.groupby(attr)
                .agg(
                    n=("question_id", "size"),
                    accuracy=("is_correct_num", "mean"),
                )
                .reset_index()
                .rename(columns={attr: "group"})
            )
            valid = group_perf[group_perf["n"] >= MIN_BUCKET_SUPPORT].copy()
            if len(valid) >= 2:
                best_row = valid.sort_values("accuracy", ascending=False).iloc[0]
                worst_row = valid.sort_values("accuracy", ascending=True).iloc[0]
                gap = float(best_row["accuracy"] - worst_row["accuracy"])
                path_rows.append(
                    {
                        "attribute": attr,
                        "plan_bucket": bucket,
                        "n_bucket": int(len(bdf)),
                        "groups_with_support": int(len(valid)),
                        "best_group": best_row["group"],
                        "worst_group": worst_row["group"],
                        "best_acc": float(best_row["accuracy"]),
                        "worst_acc": float(worst_row["accuracy"]),
                        "bucket_gap": gap,
                    }
                )

            per_group_bucket = valid.set_index("group")["accuracy"].to_dict()
            wb = float(bucket_weights.get(bucket, 0.0))
            for g in std_acc.keys():
                std_acc[g] += wb * float(per_group_bucket.get(g, group_means.get(g, np.nan)))

        std_df = pd.DataFrame(
            [{"attribute": attr, "group": g, "standardized_accuracy": float(a)} for g, a in std_acc.items()]
        )
        if not std_df.empty:
            std_df = std_df.sort_values("standardized_accuracy")
            path_std_rows.extend(std_df.to_dict("records"))
            raw_group_acc = (
                attr_df.groupby(attr)["is_correct_num"]
                .mean()
                .reset_index()
                .rename(columns={attr: "group", "is_correct_num": "raw_accuracy"})
            )
            merged = std_df.merge(raw_group_acc, on="group", how="left")
            raw_gap = float(raw_group_acc["raw_accuracy"].max() - raw_group_acc["raw_accuracy"].min()) if len(raw_group_acc) >= 2 else np.nan
            std_gap = (
                float(merged["standardized_accuracy"].max() - merged["standardized_accuracy"].min())
                if len(merged) >= 2
                else np.nan
            )
            path_summary_rows.append(
                {
                    "attribute": attr,
                    "n": int(len(attr_df)),
                    "n_groups": int(attr_df[attr].nunique()),
                    "raw_gap": raw_gap,
                    "path_standardized_gap": std_gap,
                    "gap_reduction": (raw_gap - std_gap) if pd.notna(raw_gap) and pd.notna(std_gap) else np.nan,
                }
            )

    path_df = pd.DataFrame(
        path_rows,
        columns=[
            "attribute",
            "plan_bucket",
            "n_bucket",
            "groups_with_support",
            "best_group",
            "worst_group",
            "best_acc",
            "worst_acc",
            "bucket_gap",
        ],
    )
    path_df.to_csv(out_dir / "lens_agentic_path_conditioned_gap.csv", index=False)

    path_std_df = pd.DataFrame(path_std_rows, columns=["attribute", "group", "standardized_accuracy"])
    path_std_df.to_csv(out_dir / "lens_agentic_counterfactual_policy_eval.csv", index=False)
    path_summary_df = pd.DataFrame(
        path_summary_rows,
        columns=["attribute", "n", "n_groups", "raw_gap", "path_standardized_gap", "gap_reduction"],
    )
    path_summary_df.to_csv(out_dir / "lens_agentic_counterfactual_policy_summary.csv", index=False)

    for attr in sensitive_attrs:
        sub = path_df[path_df["attribute"] == attr].sort_values("bucket_gap", ascending=False).head(20) if not path_df.empty else pd.DataFrame()
        if not sub.empty:
            save_bar_plot(
                sub,
                x="plan_bucket",
                y="bucket_gap",
                title=f"Path-Conditioned Accuracy Gap ({attr})",
                out_path=fig_dir / f"lens_agentic_path_conditioned_gap_{attr}.png",
                rotation=45,
            )
        std_sub = path_std_df[path_std_df["attribute"] == attr] if not path_std_df.empty else pd.DataFrame()
        if not std_sub.empty:
            save_bar_plot(
                std_sub.sort_values("standardized_accuracy", ascending=False),
                x="group",
                y="standardized_accuracy",
                title=f"Counterfactual Planner-Standardized Accuracy ({attr})",
                out_path=fig_dir / f"lens_agentic_counterfactual_policy_{attr}.png",
                rotation=25,
            )

    # Lens 2b: Transition disparity (tool-order divergence)
    transition_rows = []
    jsd_rows = []
    for attr in sensitive_attrs:
        attr_df = df[df[attr].notna()].copy()
        if attr_df.empty:
            continue
        attr_df[attr] = attr_df[attr].astype(str)
        attr_df = attr_df[attr_df[attr] != "UNKNOWN"]
        if attr_df.empty:
            continue

        per_group_counts: Dict[str, Counter] = {}
        all_transitions = set()
        for grp, gdf in attr_df.groupby(attr, dropna=False):
            c = Counter()
            n_transitions = 0
            for seq in gdf["tool_sequence"].fillna("NONE").astype(str):
                toks = [t for t in seq.split("->") if t and t != "NONE"]
                if not toks:
                    continue
                prev = "START"
                for t in toks:
                    tr = f"{prev}->{t}"
                    c[tr] += 1
                    all_transitions.add(tr)
                    n_transitions += 1
                    prev = t
            per_group_counts[str(grp)] = c
            total = sum(c.values())
            for tr, n_tr in c.items():
                transition_rows.append(
                    {
                        "attribute": attr,
                        "group": str(grp),
                        "transition": tr,
                        "count": int(n_tr),
                        "rate": float(n_tr / total) if total > 0 else np.nan,
                        "n_questions": int(len(gdf)),
                        "n_transitions": int(total),
                    }
                )

        if not all_transitions:
            continue
        tr_list = sorted(all_transitions)
        groups = sorted(per_group_counts.keys())
        for ga, gb in combinations(groups, 2):
            ca = per_group_counts.get(ga, Counter())
            cb = per_group_counts.get(gb, Counter())
            va = np.array([float(ca.get(t, 0)) for t in tr_list], dtype=float)
            vb = np.array([float(cb.get(t, 0)) for t in tr_list], dtype=float)
            if va.sum() == 0 or vb.sum() == 0:
                continue
            jsd = js_divergence(va, vb)
            n_qa = int((attr_df[attr].astype(str) == ga).sum())
            n_qb = int((attr_df[attr].astype(str) == gb).sum())
            jsd_rows.append(
                {
                    "attribute": attr,
                    "group_a": ga,
                    "group_b": gb,
                    "js_divergence": float(jsd),
                    "n_questions_a": n_qa,
                    "n_questions_b": n_qb,
                }
            )

    transition_df = pd.DataFrame(
        transition_rows,
        columns=["attribute", "group", "transition", "count", "rate", "n_questions", "n_transitions"],
    )
    transition_df.to_csv(out_dir / "lens_agentic_transition_rates_by_group.csv", index=False)

    transition_jsd_df = pd.DataFrame(
        jsd_rows,
        columns=["attribute", "group_a", "group_b", "js_divergence", "n_questions_a", "n_questions_b"],
    )
    transition_jsd_df.to_csv(out_dir / "lens_agentic_transition_divergence.csv", index=False)

    for attr in sensitive_attrs:
        sub = transition_jsd_df[transition_jsd_df["attribute"] == attr] if not transition_jsd_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        groups = sorted(set(sub["group_a"]).union(set(sub["group_b"])))
        if not groups:
            continue
        mat = pd.DataFrame(0.0, index=groups, columns=groups)
        for _, r in sub.iterrows():
            a = str(r["group_a"])
            b = str(r["group_b"])
            d = float(r["js_divergence"])
            mat.loc[a, b] = d
            mat.loc[b, a] = d
        save_matrix_plot(
            mat,
            title=f"Transition Distribution Divergence (JSD) by {attr}",
            out_path=fig_dir / f"lens_agentic_transition_divergence_{attr}.png",
        )

    # Lens 2c: Conditional tool utility by subgroup (question-type adjusted)
    utility_rows = []
    utility_gap_rows = []
    for attr in sensitive_attrs:
        attr_df = df[df[attr].notna()].copy()
        attr_df[attr] = attr_df[attr].astype(str)
        attr_df = attr_df[attr_df[attr] != "UNKNOWN"]
        if attr_df.empty:
            continue
        groups = sorted(attr_df[attr].unique().tolist())
        for t in tool_list:
            col = f"tool_used__{t}"
            group_vals = []
            for grp in groups:
                gdf = attr_df[attr_df[attr] == grp]
                used = gdf[gdf[col] == 1]
                not_used = gdf[gdf[col] == 0]
                if len(used) < MIN_GROUP_SUPPORT or len(not_used) < MIN_GROUP_SUPPORT:
                    utility_rows.append(
                        {
                            "attribute": attr,
                            "group": grp,
                            "tool": t,
                            "n_total": int(len(gdf)),
                            "n_used": int(len(used)),
                            "n_not_used": int(len(not_used)),
                            "unadjusted_uplift": np.nan,
                            "qtype_adjusted_uplift": np.nan,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "n_qtypes_supported": 0,
                            "support_ok": 0,
                        }
                    )
                    continue

                unadj = float(used["is_correct_num"].mean() - not_used["is_correct_num"].mean())
                qrows = []
                for qtype, qdf in gdf.groupby("question_type", dropna=False):
                    qu = qdf[qdf[col] == 1]
                    qn = qdf[qdf[col] == 0]
                    if len(qu) < 3 or len(qn) < 3:
                        continue
                    qrows.append((len(qdf), float(qu["is_correct_num"].mean() - qn["is_correct_num"].mean())))
                if qrows:
                    total_q = float(sum(nq for nq, _ in qrows))
                    qadj = float(sum((nq / total_q) * diff for nq, diff in qrows))
                else:
                    qadj = np.nan

                ci_low, ci_high = bootstrap_diff_ci(
                    used["is_correct_num"].to_numpy(dtype=float),
                    not_used["is_correct_num"].to_numpy(dtype=float),
                )
                row = {
                    "attribute": attr,
                    "group": grp,
                    "tool": t,
                    "n_total": int(len(gdf)),
                    "n_used": int(len(used)),
                    "n_not_used": int(len(not_used)),
                    "unadjusted_uplift": unadj,
                    "qtype_adjusted_uplift": qadj,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_qtypes_supported": int(len(qrows)),
                    "support_ok": 1,
                }
                utility_rows.append(row)
                if pd.notna(qadj):
                    group_vals.append((grp, qadj))

            if len(group_vals) >= 2:
                best_grp, best_val = max(group_vals, key=lambda x: x[1])
                worst_grp, worst_val = min(group_vals, key=lambda x: x[1])
                utility_gap_rows.append(
                    {
                        "attribute": attr,
                        "tool": t,
                        "best_group": best_grp,
                        "worst_group": worst_grp,
                        "best_qtype_adjusted_uplift": float(best_val),
                        "worst_qtype_adjusted_uplift": float(worst_val),
                        "uplift_gap": float(best_val - worst_val),
                    }
                )

    utility_df = pd.DataFrame(
        utility_rows,
        columns=[
            "attribute",
            "group",
            "tool",
            "n_total",
            "n_used",
            "n_not_used",
            "unadjusted_uplift",
            "qtype_adjusted_uplift",
            "ci_low",
            "ci_high",
            "n_qtypes_supported",
            "support_ok",
        ],
    )
    utility_df.to_csv(out_dir / "lens_agentic_conditional_tool_utility.csv", index=False)

    utility_gap_df = pd.DataFrame(
        utility_gap_rows,
        columns=[
            "attribute",
            "tool",
            "best_group",
            "worst_group",
            "best_qtype_adjusted_uplift",
            "worst_qtype_adjusted_uplift",
            "uplift_gap",
        ],
    )
    if not utility_gap_df.empty:
        utility_gap_df = utility_gap_df.sort_values(["attribute", "uplift_gap"], ascending=[True, False])
    utility_gap_df.to_csv(out_dir / "lens_agentic_conditional_tool_utility_gap.csv", index=False)

    for attr in sensitive_attrs:
        sub = utility_gap_df[utility_gap_df["attribute"] == attr].head(15) if not utility_gap_df.empty else pd.DataFrame()
        if not sub.empty:
            save_bar_plot(
                sub,
                x="tool",
                y="uplift_gap",
                title=f"Conditional Tool Utility Gap ({attr})",
                out_path=fig_dir / f"lens_agentic_conditional_tool_utility_gap_{attr}.png",
                rotation=30,
            )

    # Lens 2d: Tool-usage gap with bootstrap confidence intervals
    usage_ci_rows = []
    for attr in sensitive_attrs:
        attr_df = df[df[attr].notna()].copy()
        attr_df[attr] = attr_df[attr].astype(str)
        attr_df = attr_df[attr_df[attr] != "UNKNOWN"]
        if attr_df.empty:
            continue
        for t in tool_list:
            col = f"tool_used__{t}"
            rates = attr_df.groupby(attr)[col].mean().dropna()
            if len(rates) < 2:
                continue
            high_grp = str(rates.idxmax())
            low_grp = str(rates.idxmin())
            high_vals = attr_df[attr_df[attr] == high_grp][col].to_numpy(dtype=float)
            low_vals = attr_df[attr_df[attr] == low_grp][col].to_numpy(dtype=float)
            if len(high_vals) < MIN_GROUP_SUPPORT or len(low_vals) < MIN_GROUP_SUPPORT:
                continue
            ci_lo, ci_hi = bootstrap_diff_ci(high_vals, low_vals)
            usage_ci_rows.append(
                {
                    "attribute": attr,
                    "tool": t,
                    "high_group": high_grp,
                    "low_group": low_grp,
                    "high_group_rate": float(rates.loc[high_grp]),
                    "low_group_rate": float(rates.loc[low_grp]),
                    "gap": float(rates.loc[high_grp] - rates.loc[low_grp]),
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "n_high": int(len(high_vals)),
                    "n_low": int(len(low_vals)),
                    "support_ok": 1,
                }
            )

    usage_ci_df = pd.DataFrame(
        usage_ci_rows,
        columns=[
            "attribute",
            "tool",
            "high_group",
            "low_group",
            "high_group_rate",
            "low_group_rate",
            "gap",
            "ci_low",
            "ci_high",
            "n_high",
            "n_low",
            "support_ok",
        ],
    )
    if not usage_ci_df.empty:
        usage_ci_df = usage_ci_df.sort_values(["attribute", "gap"], ascending=[True, False])
    usage_ci_df.to_csv(out_dir / "lens_agentic_tool_usage_gap_ci.csv", index=False)

    for attr in sensitive_attrs:
        sub = usage_ci_df[usage_ci_df["attribute"] == attr].head(15) if not usage_ci_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        plt.figure(figsize=(11, 5))
        x = np.arange(len(sub))
        y = sub["gap"].to_numpy(dtype=float)
        yerr_low = y - sub["ci_low"].to_numpy(dtype=float)
        yerr_high = sub["ci_high"].to_numpy(dtype=float) - y
        plt.bar(x, y)
        plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", capsize=3)
        plt.xticks(x, sub["tool"].astype(str), rotation=30, ha="right")
        plt.title(f"Tool Usage Gap with 95% Bootstrap CI ({attr})")
        plt.ylabel("Usage gap (max group - min group)")
        plt.tight_layout()
        plt.savefig(fig_dir / f"lens_agentic_tool_usage_gap_ci_{attr}.png", dpi=150)
        plt.close()

    # Lens 3: LLM-driven reasoning bias
    llm_bias_rows = []
    llm_bias_detail_rows = []
    llm_cols = ["hedge_count", "certainty_count", "refusal_flag", "inconsistency_markers", "demographic_terms"]
    if "reasoning_quality_score" in df.columns:
        has_judge_scores = pd.to_numeric(df["reasoning_quality_score"], errors="coerce").notna().any()
        if has_judge_scores:
            llm_cols = ["reasoning_quality_score"] + llm_cols
    for attr in sensitive_attrs:
        td = text_df[text_df["attribute"] == attr]
        attr_raw = df[[attr] + llm_cols].copy()
        for c in llm_cols:
            tdc = td[["group", "n", c]].copy()
            tdc[c] = pd.to_numeric(tdc[c], errors="coerce")
            tdc = tdc[tdc[c].notna()].copy()
            for _, rr in tdc.iterrows():
                llm_bias_detail_rows.append(
                    {
                        "attribute": attr,
                        "group": rr["group"],
                        "metric": c,
                        "value": float(rr[c]),
                        "n": int(rr["n"]) if pd.notna(rr["n"]) else np.nan,
                    }
                )
            vals = tdc[c]
            if vals.notna().any() and not tdc.empty:
                max_ix = vals.idxmax()
                min_ix = vals.idxmin()
                by_group_values: Dict[str, np.ndarray] = {}
                for grp, gsub in attr_raw.groupby(attr, dropna=False):
                    gvals = pd.to_numeric(gsub[c], errors="coerce").to_numpy(dtype=float)
                    gvals = gvals[~np.isnan(gvals)]
                    if len(gvals) > 0:
                        by_group_values[str(grp)] = gvals
                ci_low, ci_high, boot_std = bootstrap_multigroup_gap_summary(by_group_values, n_boot=BOOTSTRAP_SAMPLES)
                llm_bias_rows.append(
                    {
                        "attribute": attr,
                        "metric": c,
                        "max_group": str(tdc.loc[max_ix, "group"]),
                        "max": float(vals.max()),
                        "min_group": str(tdc.loc[min_ix, "group"]),
                        "min": float(vals.min()),
                        "abs_gap": float(vals.max() - vals.min()),
                        "bootstrap_std": boot_std,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "bootstrap_n": BOOTSTRAP_SAMPLES,
                    }
                )
    llm_bias_df = pd.DataFrame(
        llm_bias_rows,
        columns=[
            "attribute",
            "metric",
            "max_group",
            "max",
            "min_group",
            "min",
            "abs_gap",
            "bootstrap_std",
            "ci_low",
            "ci_high",
            "bootstrap_n",
        ],
    )
    if not llm_bias_df.empty:
        llm_bias_df = llm_bias_df.sort_values(["attribute", "abs_gap"], ascending=[True, False])
    llm_bias_df.to_csv(out_dir / "lens_llm_reasoning_bias.csv", index=False)
    llm_bias_detail_df = pd.DataFrame(
        llm_bias_detail_rows,
        columns=["attribute", "group", "metric", "value", "n"],
    )
    if not llm_bias_detail_df.empty:
        llm_bias_detail_df = llm_bias_detail_df.sort_values(
            ["attribute", "metric", "value"],
            ascending=[True, True, False],
        )
    llm_bias_detail_df.to_csv(out_dir / "lens_llm_reasoning_bias_by_group.csv", index=False)

    for attr in sensitive_attrs:
        td = text_df[text_df["attribute"] == attr].copy()
        if not td.empty:
            plot_cols = ["hedge_count", "certainty_count", "refusal_flag"]
            if "reasoning_quality_score" in td.columns and td["reasoning_quality_score"].notna().any():
                plot_cols = ["reasoning_quality_score"] + plot_cols
            plot_td = td.set_index("group")[plot_cols]
            save_grouped_bar_from_pivot(
                plot_td,
                title=f"LLM Reasoning Bias Proxy by Group ({attr})",
                y_label="Mean feature value",
                out_path=fig_dir / f"lens_llm_reasoning_bias_{attr}.png",
            )

    # Lens 4: Modular design enables fairness mitigation (opportunity ranking)
    mitigation_rows = []
    for attr in sensitive_attrs:
        tu_attr = tool_usage_df[tool_usage_df["attribute"] == attr].set_index("group")
        st_attr = strat_df[strat_df["attribute"] == attr]
        for _, r in st_attr.iterrows():
            tool = r["tool"]
            grp = r["group"]
            use_col = f"tool_used_rate__{tool}"
            usage_rate = np.nan
            if grp in tu_attr.index and use_col in tu_attr.columns:
                usage_rate = float(pd.to_numeric(tu_attr.loc[grp, use_col], errors="coerce"))
            acc_diff = pd.to_numeric(r["acc_diff_used_minus_not"], errors="coerce")
            potential_gain = np.nan
            if pd.notna(acc_diff) and pd.notna(usage_rate):
                potential_gain = float(max(0.0, -float(acc_diff)) * float(usage_rate))
            mitigation_rows.append(
                {
                    "attribute": attr,
                    "group": grp,
                    "tool": tool,
                    "usage_rate": usage_rate,
                    "acc_diff_used_minus_not": float(acc_diff) if pd.notna(acc_diff) else np.nan,
                    "mitigation_priority_score": potential_gain,
                }
            )
    mitigation_df = pd.DataFrame(
        mitigation_rows,
        columns=["attribute", "group", "tool", "usage_rate", "acc_diff_used_minus_not", "mitigation_priority_score"],
    )
    if not mitigation_df.empty:
        mitigation_df = mitigation_df.sort_values("mitigation_priority_score", ascending=False)
    mitigation_df.to_csv(out_dir / "lens_modular_mitigation_opportunities.csv", index=False)

    for attr in sensitive_attrs:
        sub = mitigation_df[mitigation_df["attribute"] == attr].dropna(subset=["mitigation_priority_score"]).head(20)
        if not sub.empty:
            sub = sub.copy()
            sub["tool_group"] = sub["tool"].astype(str) + " | " + sub["group"].astype(str)
            save_bar_plot(
                sub,
                x="tool_group",
                y="mitigation_priority_score",
                title=f"Modular Mitigation Opportunities ({attr})",
                out_path=fig_dir / f"lens_modular_mitigation_{attr}.png",
                rotation=45,
            )

    # Lens 5: Traceable reasoning supports fairness audits
    trace_rows = []
    for attr in sensitive_attrs:
        g = df.groupby(attr, dropna=False).agg(
            n=("question_id", "size"),
            mean_tool_calls=("tool_call_count", "mean"),
            mean_unique_tools=("unique_tool_count", "mean"),
            failure_any_rate=("tool_failure_any", "mean"),
            mean_hedge=("hedge_count", "mean"),
        ).reset_index().rename(columns={attr: "group"})
        g["attribute"] = attr
        trace_rows.extend(g.to_dict("records"))
    trace_df = pd.DataFrame(
        trace_rows,
        columns=["group", "n", "mean_tool_calls", "mean_unique_tools", "failure_any_rate", "mean_hedge", "attribute"],
    )
    trace_df.to_csv(out_dir / "lens_traceable_reasoning_audit.csv", index=False)

    for attr in sensitive_attrs:
        sub = trace_df[trace_df["attribute"] == attr]
        if not sub.empty:
            piv = sub.set_index("group")[["mean_tool_calls", "failure_any_rate", "mean_hedge"]]
            save_grouped_bar_from_pivot(
                piv,
                title=f"Traceability Audit Features by Group ({attr})",
                y_label="Mean value",
                out_path=fig_dir / f"lens_traceable_audit_{attr}.png",
            )

    # Lens 6: Interaction bias (prompt sensitivity + missing context)
    interaction_summary = []
    if not prompt_df.empty:
        p_gap = float(pd.to_numeric(prompt_df["accuracy"], errors="coerce").max() - pd.to_numeric(prompt_df["accuracy"], errors="coerce").min())
        interaction_summary.append({"metric": "prompt_accuracy_gap", "value": p_gap})
    if not miss_df.empty:
        m_gap = float(pd.to_numeric(miss_df["accuracy"], errors="coerce").max() - pd.to_numeric(miss_df["accuracy"], errors="coerce").min())
        interaction_summary.append({"metric": "missing_context_accuracy_gap", "value": m_gap})
    interaction_df = pd.DataFrame(interaction_summary, columns=["metric", "value"])
    interaction_df.to_csv(out_dir / "lens_interaction_bias_summary.csv", index=False)

    if not interaction_df.empty:
        save_bar_plot(
            interaction_df,
            x="metric",
            y="value",
            title="Interaction Bias Summary Gaps",
            out_path=fig_dir / "lens_interaction_bias_summary.png",
            rotation=20,
        )

    # Lens-level compact report mapped to requested fairness categories
    lens_report = []
    lens_report.append("# Fairness Lens Visualization Report")
    lens_report.append("")
    lens_report.append("## Inherited Bias From Tools")
    lens_report.append("- Data: lens_inherited_tool_bias.csv")
    lens_report.append("- Figures: lens_inherited_tool_bias_gender_norm.png, lens_inherited_tool_bias_age_group.png")
    lens_report.append("")
    lens_report.append("## Agentic Bias Via Tool Selection")
    lens_report.append("- Data: lens_agentic_tool_selection_bias.csv, lens_agentic_path_conditioned_gap.csv, lens_agentic_counterfactual_policy_eval.csv, lens_agentic_counterfactual_policy_summary.csv")
    lens_report.append("- Data: lens_agentic_transition_rates_by_group.csv, lens_agentic_transition_divergence.csv, lens_agentic_conditional_tool_utility.csv, lens_agentic_conditional_tool_utility_gap.csv, lens_agentic_tool_usage_gap_ci.csv")
    lens_report.append("- Figures: lens_agentic_selection_bias_gender_norm.png, lens_agentic_selection_bias_age_group.png")
    lens_report.append("- Figures: lens_agentic_path_conditioned_gap_gender_norm.png, lens_agentic_path_conditioned_gap_age_group.png")
    lens_report.append("- Figures: lens_agentic_counterfactual_policy_gender_norm.png, lens_agentic_counterfactual_policy_age_group.png")
    lens_report.append("- Figures: lens_agentic_transition_divergence_gender_norm.png, lens_agentic_transition_divergence_age_group.png")
    lens_report.append("- Figures: lens_agentic_conditional_tool_utility_gap_gender_norm.png, lens_agentic_conditional_tool_utility_gap_age_group.png")
    lens_report.append("- Figures: lens_agentic_tool_usage_gap_ci_gender_norm.png, lens_agentic_tool_usage_gap_ci_age_group.png")
    lens_report.append("")
    lens_report.append("## LLM-Driven Reasoning Bias")
    lens_report.append("- Data: lens_llm_reasoning_bias.csv")
    lens_report.append("- Figures: lens_llm_reasoning_bias_gender_norm.png, lens_llm_reasoning_bias_age_group.png")
    lens_report.append("")
    lens_report.append("## Modular Design Enables Fairness Mitigation")
    lens_report.append("- Data: lens_modular_mitigation_opportunities.csv")
    lens_report.append("- Figures: lens_modular_mitigation_gender_norm.png, lens_modular_mitigation_age_group.png")
    lens_report.append("")
    lens_report.append("## Traceable Reasoning Supports Fairness Audits")
    lens_report.append("- Data: lens_traceable_reasoning_audit.csv")
    lens_report.append("- Figures: lens_traceable_audit_gender_norm.png, lens_traceable_audit_age_group.png")
    lens_report.append("")
    lens_report.append("## Interaction Bias")
    lens_report.append("- Data: prompt_sensitivity_summary.csv, missing_context_summary.csv, lens_interaction_bias_summary.csv")
    lens_report.append("- Figures: prompt_sensitivity_accuracy.png, missing_context_accuracy.png, lens_interaction_bias_summary.png")
    lens_report.append("")
    lens_report.append("## Important Note")
    lens_report.append("- Prompt sensitivity results are observational from existing logs, not counterfactual prompt-rewrite experiments.")
    (out_dir / "fairness_lens_report.md").write_text("\n".join(lens_report))

    if paper_figures_only:
        write_paper_figures_only_outputs(
            out_dir=out_dir,
            log_path=log_path,
            meta_q_path=meta_q_path,
            dataset_label=paper_dataset_label,
            model_label=paper_model_label,
        )


def run_batch(
    input_root: Path,
    out_root: Path,
    meta_q_path: Path,
    meta_case_path: Path,
    keep_think_text: bool = False,
    enable_llm_judge: bool = False,
    judge_model: str = "deepseek-chat",
    judge_base_url: str = "https://api.deepseek.com/v1",
    judge_api_key_env: str = "DEEPSEEK_API_KEY",
    judge_max_samples: Optional[int] = None,
    judge_concurrency: int = 10,
    paper_figures_only: bool = False,
    paper_dataset_label: Optional[str] = None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_dirs = [p for p in input_root.iterdir() if p.is_dir() and p.name != "analysis"] # [PosixPath('./logs/chexagentbench/qwen38b-vllm'), PosixPath('./logs/chexagentbench/qwen3vl8b-vllm'), PosixPath('./logs/chexagentbench/mistral-7b-vllm'), PosixPath('./logs/chexagentbench/llama-3.1-8b-vllm')]
    if not baseline_dirs:
        print(f"No baseline directories found under {input_root}")
        return

    for baseline_dir in sorted(baseline_dirs):
        baseline_name = baseline_dir.name # e.g., 'llama-3.1-8b-vllm'
        log_path = pick_baseline_log(baseline_dir, baseline_name)
        if not log_path:
            print(f"No suitable log file found for baseline: {baseline_name}") # e.g., PosixPath('./logs/chexagentbench/llama-3.1-8b-vllm/llama-3.1-8b-vllm_20260203_110431.json')
            continue
        out_dir = out_root / baseline_name / "fairness_posthoc"
        print(f"[fairness] baseline={baseline_name}")
        print(f"[fairness] input={log_path}")
        print(f"[fairness] output={out_dir}")
        run_single(
            log_path,
            out_dir,
            meta_q_path,
            meta_case_path,
            keep_think_text=keep_think_text,
            enable_llm_judge=enable_llm_judge,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key_env=judge_api_key_env,
            judge_max_samples=judge_max_samples,
            judge_concurrency=judge_concurrency,
            paper_figures_only=paper_figures_only,
            paper_dataset_label=paper_dataset_label,
            paper_model_label=baseline_name,
        ) # meta_q: PosixPath('./data/chestagentbench/metadata.jsonl') meta_case: PosixPath('./data/eurorad_metadata.json')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-path",
        default=str(LOG_PATH),
        help="Path to the input log file (JSONL).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory to save outputs for single-log analysis.",
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="Root directory containing baseline subdirectories to analyze in batch.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Root directory to store batch analysis outputs (one folder per baseline).",
    )
    parser.add_argument(
        "--meta-q-path",
        default=str(META_Q_PATH),
        help="Question metadata path (JSONL), e.g. data/mimic/medrax_input_all_2000.jsonl.",
    )
    parser.add_argument(
        "--meta-case-path",
        default=str(META_CASE_PATH),
        help="Case metadata path (JSON or CSV), e.g. data/mimic/mimic_sample_400.csv.",
    )
    parser.add_argument(
        "--keep-think-text",
        action="store_true",
        default=True,
        help="If set, compute text features on full model text including <think>...</think> blocks.",
    )
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="If set, score per-question reasoning quality with DeepSeek and include subgroup gap metrics.",
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek-chat",
        help="Judge model name for OpenAI-compatible API.",
    )
    parser.add_argument(
        "--judge-base-url",
        default="https://api.deepseek.com/v1",
        help="Base URL for judge model API.",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable name that stores judge API key.",
    )
    parser.add_argument(
        "--judge-max-samples",
        type=int,
        default=None,
        help="Optional cap on number of rows to judge per run (for quick dry-runs).",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=10,
        help="Number of concurrent DeepSeek judge requests.",
    )
    parser.add_argument(
        "--paper-figures-only",
        action="store_true",
        help=(
            "Strict release mode: after reading the log, keep only paper Fig. 2/Fig. 3 "
            "outputs plus their backing CSV data and manifest."
        ),
    )
    parser.add_argument(
        "--paper-dataset-label",
        default=None,
        help="Optional dataset label for paper figures, e.g. CheXAgentBench or MIMIC-FairnessVQA.",
    )
    parser.add_argument(
        "--paper-model-label",
        default=None,
        help="Optional model label for paper figures, e.g. Gemini3.",
    )
    args = parser.parse_args()
    meta_q_path = Path(args.meta_q_path)
    meta_case_path = Path(args.meta_case_path)

    if args.input_root or args.out_root:
        input_root = Path(args.input_root or "./logs/chexagentbench")
        out_root = Path(args.out_root or "./logs/chexagentbench/analysis")
        run_batch(
            input_root,
            out_root,
            meta_q_path,
            meta_case_path,
            keep_think_text=args.keep_think_text,
            enable_llm_judge=args.enable_llm_judge,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            judge_api_key_env=args.judge_api_key_env,
            judge_max_samples=args.judge_max_samples,
            judge_concurrency=args.judge_concurrency,
            paper_figures_only=args.paper_figures_only,
            paper_dataset_label=args.paper_dataset_label,
        )
    else:
        run_single(
            Path(args.log_path),
            Path(args.out_dir),
            meta_q_path,
            meta_case_path,
            keep_think_text=args.keep_think_text,
            enable_llm_judge=args.enable_llm_judge,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            judge_api_key_env=args.judge_api_key_env,
            judge_max_samples=args.judge_max_samples,
            judge_concurrency=args.judge_concurrency,
            paper_figures_only=args.paper_figures_only,
            paper_dataset_label=args.paper_dataset_label,
            paper_model_label=args.paper_model_label,
        )
