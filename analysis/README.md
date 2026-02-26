# Analysis Metrics and Outputs Guide

This document explains all analysis scripts in `analysis/`, what each metric/output means, how each one evaluates subgroup fairness, and how to interpret results.

## Scope

Scripts covered:
- `analysis/fairness_posthoc.py`
- `analysis/compute_fairness_metrics.py`
- `analysis/decompose_bias_from_fairness.py`
- `analysis/generate_mitigation_config.py`
- `analysis/analyze_tool_usage.py`

Sensitive subgroup attributes used by fairness analyses:
- `gender_norm`
- `age_group`
- intersection: `gender_norm|age_group` (in selected outputs)

CheXAgentBench-specific rule in `analysis/fairness_posthoc.py`:
- `age_group` is forced to two buckets only: `old` (age `>= 60`) and `young` (age `< 60`).

## Recommended End-to-End Pipeline

1. Run `analysis/fairness_posthoc.py` to create subgroup-level fairness artifacts.
2. Run `analysis/compute_fairness_metrics.py` for compact classic fairness metrics (ACC, DP, EoD, etc.).
3. Run `analysis/decompose_bias_from_fairness.py` to separate raw gap into planning/reasoning/residual components.
4. Run `analysis/generate_mitigation_config.py` to generate actionable mitigation policy YAML.
5. Optionally run `analysis/analyze_tool_usage.py` for operational tool diagnostics (not a direct fairness metric, but useful context).

## 1) `fairness_posthoc.py`: Main Fairness Analysis

### What it does

Builds a per-question feature table from logs + metadata, then creates fairness tables and figures across six lens categories:
- Lens 1: inherited tool bias
- Lens 2: agentic tool-selection and trajectory bias
- Lens 3: LLM reasoning-style bias
- Lens 4: modular mitigation opportunities
- Lens 5: traceability/audit readiness
- Lens 6: interaction bias (prompt/context effects)

It is observational and post-hoc. It quantifies disparity signals; it does not prove causality.

### Key thresholds used

- `MIN_GROUP_SUPPORT = 10`
- `MIN_BUCKET_SUPPORT = 8`

Interpretation: small-group estimates are unstable; low-support rows should not drive strong decisions.

### Core output directory

Per baseline:
- `<out_root>/<baseline>/fairness_posthoc/`

Single run:
- `--out-dir` path

### Output dictionary: files, metrics, fairness meaning, interpretation

#### Data foundation

- `per_question_features.csv`
  - One row per `question_id`.
  - Contains outcome (`is_correct`), demographics (`gender_norm`, `age_group`), tool features (`tool_call_count`, `first_tool`, `tool_sequence`, `tool_used__*`, `tool_failed__*`), text features (`hedge_count`, `certainty_count`, `refusal_flag`, etc.), context completeness flags, and prompt buckets.
  - Fairness role: this is the canonical feature store used by all downstream subgroup metrics.
  - Important clarification on correctness:
    - `is_correct` is defined at the single-question level (0/1 per row).
    - Subgroup correctness is computed by aggregation, not relabeling:
      - `subgroup_accuracy = mean(is_correct)` over rows in that subgroup.
    - This is standard for fairness analysis: outcomes are per example, disparities are properties of group-level aggregates.

- `coverage.json`
  - Fields: `total_questions`, matched demographic counts, prompt explicit counts, missing-context counts, status quality.
  - Fairness role: coverage audit. If subgroup metadata match rates are low or imbalanced, subgroup fairness conclusions are weaker.

#### Performance disparity tables

- `group_performance.csv`
  - Columns: `attribute`, `group`, `n`, `accuracy`, `bootstrap_std`, `ci_low`, `ci_high`, `bootstrap_n`.
  - Metric: subgroup accuracy with non-parametric bootstrap (1000 resamples) standard deviation and CI.
  - Fairness meaning: direct subgroup outcome disparity.
  - Interpretation: compare worst vs best group; prioritize gaps with CI separation and adequate `n`.

- `group_performance_intersectional.csv`
  - Same metric at intersection `gender_norm|age_group`, including `bootstrap_std`, `ci_low`, `ci_high`, `bootstrap_n` (1000).
  - Fairness meaning: detects disparities hidden in marginal (single-attribute) analysis.
  - Interpretation: helps reveal concentrated harms in specific intersections.

- `group_performance_by_question_type.csv`
  - Adds `question_type`.
  - Includes `bootstrap_std`, `ci_low`, `ci_high`, `bootstrap_n` (1000) for each subgroup-question_type slice.
  - Fairness meaning: whether subgroup gaps are task-dependent.
  - Interpretation: if disparity appears in only some question types, mitigation should be targeted there.

#### Tool behavior by subgroup

- `tool_usage_by_group.csv`
  - Columns include `tool_call_rate`, `tool_used_rate__<tool>`, `tool_fail_rate__<tool>`.
  - Construction from per-question rows (within each subgroup):
    - `tool_call_rate = mean(tool_call_count)`
    - `tool_used_rate__t = mean(1{tool t used at least once in the question trace})`
    - `tool_fail_rate__t = mean(tool_failed__t)`, where `tool_failed__t` is failed-call count for tool `t` in that question.
      - If a tool is called at most once per question, this behaves like a failure probability.
      - If a tool can fail multiple times in one question, this is expected failed calls per question.
  - Fairness meaning: exposure parity and failure parity across groups.
  - Interpretation: high usage gaps indicate policy/exposure skew; high failure-rate gaps indicate inherited reliability inequity.

- `first_tool_by_group.csv`
  - `first_tool` frequency by subgroup.
  - Fairness meaning: front-door planning differences.
  - Interpretation: first-tool divergence often cascades into downstream performance differences.

- `tool_sequences_top10_by_group.csv`
  - Top trajectories per subgroup.
  - Fairness meaning: high-level behavior differences in full agent pathways.

- `outcome_by_tool_usage_within_group.csv`
  - Columns: `n_used`, `n_not_used`, `acc_used`, `acc_not_used`, `acc_diff_used_minus_not`.
  - Metric: within-group tool utility proxy for each tool.
    - For subgroup `g` and tool `t`, split rows into:
      - `used`: rows with `tool_used__t == 1`
      - `not_used`: rows with `tool_used__t == 0`
    - Compute:
      - `acc_used = mean(is_correct | used)`
      - `acc_not_used = mean(is_correct | not_used)`
      - `acc_diff_used_minus_not = acc_used - acc_not_used`
    - Positive values suggest observational uplift; negative values suggest observational harm.
  - Fairness meaning: whether a tool helps one group but harms another.
  - Interpretation: negative `acc_diff_used_minus_not` suggests potential subgroup harm when that tool is used.

#### Prompt/context interaction tables

- `prompt_sensitivity_summary.csv`
  - By `prompt_bucket` (`demographic_explicit` vs `neutral`), with accuracy CI and behavior rates.
  - Fairness meaning: observational sensitivity to demographic wording in prompts.

- `prompt_sensitivity_by_group.csv`
  - Prompt bucket performance split by subgroup.
  - Fairness meaning: whether prompt style changes groups differently.

- `missing_context_summary.csv`
  - By context completeness bucket (`full`, `medium`, `low`, `unknown`) with accuracy CI.
  - Fairness meaning: robustness parity under incomplete context.

- `missing_context_by_group.csv`
  - Context-bucket performance by subgroup.
  - Fairness meaning: whether context incompleteness disproportionately impacts some groups.

#### LLM textual style table

- `llm_text_features_by_group.csv`
  - Means for `hedge_count`, `certainty_count`, `refusal_flag`, `demographic_terms`, `inconsistency_markers`, etc.
  - Fairness meaning: subgroup-dependent style or caution shifts in model outputs.
  - Interpretation: large style gaps can signal uneven confidence behavior that may mediate outcome disparity.

#### Statistical association table

- `association_tests.csv`
  - Chi-square tests for `group vs outcome`, `group vs tool usage`, `group vs tool failure`, plus FDR-adjusted p-values (`p_fdr_bh`).
  - Fairness meaning: whether subgroup and outcomes/behavior are statistically associated.
  - Interpretation: significance is not effect size. Use with gap magnitude and support counts.

#### Narrative summaries

- `summary.md`
  - Includes heuristic attribution cues and descriptive findings.
  - Fairness meaning: human-readable synthesis only; not definitive attribution.

- `fairness_lens_report.md`
  - Maps generated files and figures by lens category.

#### Lens 1: inherited tool bias

- `lens_inherited_tool_bias.csv`
  - What this table is:
    - One row per `(attribute, tool)` pair (e.g., `gender_norm x image_visualizer`).
    - It fuses reliability disparity and utility disparity into a compact inherited-risk summary.
  - Columns:
    - Core: `tool_failure_gap`, `tool_usage_gap`, `min_acc_diff_used_minus_not`
    - Direct subgroup performance view:
      - `best_group_acc_used`, `best_group_acc_used_value`
      - `worst_group_acc_used`, `worst_group_acc_used_value`
      - `acc_used_gap`
      - For binary attributes: `acc_used_female`, `acc_used_male`, `acc_used_young`, `acc_used_old`
  - Metrics:
    - `tool_failure_gap = max_group_fail_rate - min_group_fail_rate`
    - `tool_usage_gap = max_group_usage_rate - min_group_usage_rate`
    - `min_acc_diff_used_minus_not = min_g acc_diff_used_minus_not(g, tool)` (worst subgroup utility; can be negative)
    - `acc_used_<group> = mean(is_correct | tool used, subgroup=<group>)`
    - `acc_used_gap = max_g acc_used(g, tool) - min_g acc_used(g, tool)`
  - Fairness meaning: subgroup harm inherited from tool reliability and subgroup-specific tool impact.
  - Interpretation:
    - Large `tool_failure_gap` means reliability differs by subgroup for the same tool.
    - Large negative `min_acc_diff_used_minus_not` means at least one subgroup is associated with worse outcomes when this tool is used.
    - `tool_usage_gap` is included as context for exposure, but the inherited-bias signal is mainly reliability/utility asymmetry.

#### Lens 2: agentic bias via tool selection and trajectories

- `lens_agentic_tool_selection_bias.csv`
  - Column `usage_gap` by tool and attribute.
  - Fairness meaning: exposure inequality in planner tool choices.
  - Use this lens when your question is: "Does the planner route subgroups to different tools?"

#### Which metric for which question?

- Tool selection bias (planner/exposure): start with `lens_agentic_tool_selection_bias.csv` (`usage_gap`).
- Inherited tool bias (tool reliability/utility asymmetry): start with `lens_inherited_tool_bias.csv` (`tool_failure_gap`, `min_acc_diff_used_minus_not`).
- Practical read: inherited harm risk is highest when a tool has both unfavorable inherited metrics and non-trivial subgroup exposure.

- `lens_agentic_path_conditioned_gap.csv`
  - Compares subgroup accuracy inside same `plan_bucket` (`first_tool|calls=<bin>`).
  - Key metric: `bucket_gap = best_acc - worst_acc`.
  - Fairness meaning: within-path disparity; controls coarse trajectory mix.

- `lens_agentic_counterfactual_policy_eval.csv`
  - `standardized_accuracy` per group after reweighting to common bucket mix.
  - Fairness meaning: counterfactual “same planner mix for all groups” estimate.

- `lens_agentic_counterfactual_policy_summary.csv`
  - `raw_gap`, `path_standardized_gap`, `gap_reduction`.
  - Interpretation:
    - `gap_reduction > 0`: path allocation explains part of disparity.
    - Large residual `path_standardized_gap`: disparity remains beyond planner mix.

- `lens_agentic_transition_rates_by_group.csv`
  - Transition probabilities like `START->ToolA`, `ToolA->ToolB`.
  - Fairness meaning: subgroup differences in tool-order policy.

- `lens_agentic_transition_divergence.csv`
  - Pairwise `js_divergence` between subgroup transition distributions.
  - Interpretation: near 0 means similar trajectories; larger values mean more divergent plans.

- `lens_agentic_conditional_tool_utility.csv`
  - Per subgroup/tool utility:
    - `unadjusted_uplift = Acc(used) - Acc(not_used)`
    - `qtype_adjusted_uplift` controls question-type mix
    - bootstrap CI for uplift difference
  - Fairness meaning: unequal benefit/harm of same tool across groups.

- `lens_agentic_conditional_tool_utility_gap.csv`
  - `uplift_gap = best_qtype_adjusted_uplift - worst_qtype_adjusted_uplift`.
  - Interpretation: higher gap means stronger subgroup inequality in tool utility.

- `lens_agentic_tool_usage_gap_ci.csv`
  - For each tool: max-group vs min-group usage gap + bootstrap CI.
  - Interpretation: if CI excludes 0, usage disparity is more stable.

#### Lens 3: reasoning bias

- `lens_llm_reasoning_bias.csv`
  - For each reasoning proxy metric: `max`, `min`, `abs_gap`.
  - Fairness meaning: subgroup-dependent reasoning style proxies.

#### Lens 4: mitigation opportunities

- `lens_modular_mitigation_opportunities.csv`
  - `mitigation_priority_score = max(0, -acc_diff_used_minus_not) * usage_rate`
  - Fairness meaning: prioritizes tool/group pairs where a harmful tool is frequently used.
  - Interpretation: high score = high-impact candidate for gating/fallback/prompt constraints.

#### Lens 5: traceability audit

- `lens_traceable_reasoning_audit.csv`
  - Group-level means: tool calls, unique tools, failure rate, hedge.
  - Fairness meaning: whether audit-relevant process behaviors differ by subgroup.

#### Lens 6: interaction bias summary

- `lens_interaction_bias_summary.csv`
  - `prompt_accuracy_gap` and `missing_context_accuracy_gap`.
  - Fairness meaning: summarizes interaction-channel disparities.

#### Figures

- `figures/*.png` contains visual versions of core metrics (bar charts, matrices, CI plots).
- Use figures for triage; use CSVs for exact values and decision thresholds.

## 2) `compute_fairness_metrics.py`: Compact Classical Fairness Metrics

### Input

Reads:
- `./logs/<dataset>/analysis/<llm>/fairness_posthoc/per_question_features.csv`

Uses columns:
- `predicted_answer`, `correct_answer`, `age_raw`, `gender`

Converts to binary subgroup labels:
- `age_raw`: `>=60` vs `<60`
- `gender`: male vs non-male

### Metrics produced (printed, not saved as CSV by default)

- `ACC`
  - Overall accuracy.
  - Utility baseline, not fairness by itself.

- `Delta-ACC`
  - Absolute accuracy difference between two subgroup bins.
  - Fairness meaning: outcome parity gap.
  - Lower is better.

- `DP` (Demographic Parity gap, one-vs-rest over classes, max over classes)
  - Gap in predicted-positive rates between subgroups.
  - Fairness meaning: exposure/decision-rate parity independent of label.
  - Lower is better; large values show differential positive assignment rates.

- `EoD` (Equalized Odds Difference)
  - Computes subgroup TPR and FPR gaps per class; uses max class gap; final score is average of TPR gap and FPR gap.
  - Fairness meaning: error-rate parity conditional on truth.
  - Lower is better.

- `FUT` (Fairness-Utility Tradeoff proxy)
  - `mean_acc / (1 + std_acc)` over subgroup bins.
  - Higher generally means good average performance with low subgroup spread.

### Interpretation notes

- This script is coarse and binary-group-only.
- It complements, but does not replace, the richer multi-group/lens outputs in `fairness_posthoc.py`.

## 3) `decompose_bias_from_fairness.py`: Gap Decomposition

### What it does

Consumes fairness outputs and decomposes subgroup disparity into:
- raw gap
- planning-adjusted gap
- planning+reasoning-adjusted gap
- residual gap

It also pulls proxy maxima from lens outputs and emits recommendation strings.

### Inputs expected per baseline

From `<analysis_root>/<baseline>/fairness_posthoc/`:
- `per_question_features.csv`
- `lens_inherited_tool_bias.csv`
- `lens_agentic_tool_selection_bias.csv`
- `lens_llm_reasoning_bias.csv`

### Outputs

- `bias_decomposition.csv`
- `bias_decomposition.json`
- `bias_decomposition.md`

### Main metrics

- `raw_gap`
  - Max subgroup accuracy minus min subgroup accuracy.

- `planning_adjusted_gap`
  - Gap after standardizing by plan bucket (trajectory/tool-set/call-count).

- `planning_reasoning_adjusted_gap`
  - Gap after additional reasoning bucket standardization.

- `plan_explained_gap = max(0, raw_gap - planning_adjusted_gap)`
- `reason_explained_gap = max(0, planning_adjusted_gap - planning_reasoning_adjusted_gap)`
- `residual_gap = planning_reasoning_adjusted_gap`

- Proxy severity fields:
  - `tool_failure_gap_max_abs`
  - `tool_harm_gap_max_abs`
  - `planning_usage_gap_max_abs`
  - `reasoning_feature_gap_max_abs`

### Fairness interpretation

- Large `plan_explained_gap`: planner/path allocation likely drives a meaningful part of disparity.
- Large `reason_explained_gap`: reasoning-style differences likely explain additional disparity.
- Large `residual_gap`: unexplained disparity remains after both controls; escalate with stronger audits/calibration/counterfactual tests.

## 4) `generate_mitigation_config.py`: Turn Metrics into Policy Controls

### What it does

Generates mitigation YAML from decomposition + mitigation-opportunity outputs.

### Inputs

- Decomposition CSV:
  - `.../bias_decomposition.csv`
- Per-baseline fairness opportunities:
  - `<analysis_root>/<baseline>/fairness_posthoc/lens_modular_mitigation_opportunities.csv`

### Output

- YAML (default):
  - `analysis/mitigation_config.chexagentbench.yaml`

### Decision thresholds (defaults)

- `residual_gap >= 0.02` -> enable `system_calibration`
- `planning_usage_gap_max_abs >= 0.08` or `plan_explained_gap >= 0.01` -> enable `planning_guardrails`
- `reasoning_feature_gap_max_abs >= 0.05` or `reason_explained_gap >= 0.01` -> enable `reasoning_guardrails`
- `tool_harm_gap_max_abs >= 0.08` -> enable `tool_gating`

### Fairness interpretation

This is a policy compiler, not an evaluator. It operationalizes measured subgroup disparity signals into enabled controls and tool priorities.

## 5) `analyze_tool_usage.py`: Operational Tool Diagnostics

### What it does

Parses JSON/JSONL logs and creates tool-call summaries.

### Outputs

- `tool_usage_summary.json`
  - `total_tool_calls`, `successful_calls`, `failed_calls`, `average_latency_ms`, `unique_tools_used`.
- `tool_calls.csv`
  - per-call rows: `session_id`, `tool_name`, `status`, `latency_ms`, `result`, `error_message`.
- `tool_usage_by_tool.csv`
  - by-tool counts/success/failure/latency.
- `tool_usage_by_session.csv`
  - per-session call/success/failure/latency stats.
- `report.md`
  - summary + top tools.

### Fairness relevance

No subgroup columns are used here, so this is not a direct subgroup fairness analysis. It is still useful for:
- diagnosing globally unreliable tools that may later appear in subgroup fairness gaps,
- validating operational quality before fairness attribution.

## Practical Interpretation Checklist

1. Start with `group_performance.csv` and `group_performance_intersectional.csv` to confirm outcome gaps.
2. Check `coverage.json` to ensure subgroup metadata quality is sufficient.
3. Use Lens 2 outputs to determine if planner/trajectory differences drive disparity.
4. Use Lens 1 and Lens 2c to identify tool-specific subgroup harm.
5. Use decomposition outputs to quantify explained vs residual gaps.
6. Use mitigation config YAML as the actionable control set.

## Caveats

- Observational log analysis only; not causal.
- Confidence intervals quantify sampling uncertainty, not confounding.
- Small `n` and sparse subgroup-tool cells can inflate volatility.
- Use effect size + CI + support jointly; do not rely on p-values alone.
