# DUCX: Decomposing Unfairness in Tool-Using Chest X-ray Agents

This repository contains the code release for:

> **DUCX: Decomposing Unfairness in Tool-Using Chest X-ray Agents**  
> Zikang Xu, Ruinan Jin, Xiaoxiao Li  
> arXiv: https://arxiv.org/abs/2603.00777

DUCX audits demographic fairness in MedRAX-style chest X-ray agents. The release focuses on the paper setting only: ChestAgentBench, MIMIC-FairnessVQA, five driver LLM backbones, end-to-end fairness metrics, tool-exposure bias, tool-transition bias, and LLM-reasoning bias.

## What Is Included

- Agent execution over multiple-choice chest X-ray QA datasets.
- Paper metrics: ACC, Delta-ACC, demographic parity, equalized odds, and fairness-utility tradeoff.
- DUCX decomposition:
  - tool exposure bias: subgroup utility gaps conditioned on tool use;
  - tool transition bias: subgroup differences in tool-routing transitions;
  - LLM reasoning bias: judge-score, hedging, and demographic-term gaps.
- Example MIMIC-FairnessVQA-format JSONL data.
- A Colab-ready quickstart notebook in [notebooks/DUCX_quickstart.ipynb](notebooks/DUCX_quickstart.ipynb), with executable setup, dry-run, agent-run, and analysis cells.

Exploratory utilities inherited from the upstream MedRAX/ChestAgentBench codebase are kept under [experiments/](experiments/) but are not part of the default DUCX paper reproduction path.

## Environment

Use conda to create an isolated Python environment:

```bash
conda env create -f environment.yml
conda activate ducx
```

Optional dependencies:

```bash
# Local OpenAI-compatible LLM serving.
pip install vllm

# Native Gemini tool-calling backend.
pip install -e ".[gemini]"
```

The full agent uses GPU-backed chest X-ray tools.

## Data

Create the expected local directories:

```bash
mkdir -p data/chestagentbench data/mimic figures logs model-weights temp
```

ChestAgentBench / EuroRAD metadata is provided by MedRAX and can be downloaded from https://github.com/bowang-lab/MedRAX/tree/main/data. Expected files:

- `data/chestagentbench/metadata.jsonl`
- `data/eurorad_metadata.json`
- image files whose paths resolve from the `images` field

MIMIC-FairnessVQA is provided by this DUCX paper release. Expected files:

- `data/mimic/medrax_input_all_2000.jsonl`
- MIMIC demographic metadata, for example `data/mimic/mimic_sample_400.csv`
- image files whose paths resolve from the `images` field

See [data/README.md](data/README.md) for provenance, download links, and the expected schema. The repository includes [data/mimic-fairnessVQA_example.jsonl](data/mimic-fairnessVQA_example.jsonl) as a format example only; it does not include MIMIC-CXR images.

## Run Agent Evaluation

Start an OpenAI-compatible local server, for example Qwen3-VL-8B:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Set the agent LLM endpoint:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
export OPENAI_MODEL="qwen3-vl-8b"
```

Run ChestAgentBench:

```bash
python launch_over_chexbench.py \
  --model "${OPENAI_MODEL}" \
  --model-dir ./model-weights \
  --temp-dir ./temp \
  --data-file data/chestagentbench/metadata.jsonl \
  --device cuda \
  --log-prefix qwen3vl8b-vllm \
  --llm-parse
```

Run MIMIC-FairnessVQA:

```bash
python launch_over_chexbench.py \
  --model "${OPENAI_MODEL}" \
  --model-dir ./model-weights \
  --temp-dir ./temp \
  --data-file data/mimic/medrax_input_all_2000.jsonl \
  --device cuda \
  --log-prefix qwen3vl8b-vllm-mimic \
  --llm-parse
```

Logs are written under `logs/<log-prefix>/`.

## Paper Analysis

Analyze one ChestAgentBench run:

```bash
python analysis/fairness_posthoc.py \
  --log-path logs/qwen3vl8b-vllm/<run_log>.json \
  --meta-q-path data/chestagentbench/metadata.jsonl \
  --meta-case-path data/eurorad_metadata.json \
  --out-dir logs/chexagentbench/analysis/qwen3vl8b-vllm/fairness_posthoc \
  --enable-llm-judge
```

Analyze one MIMIC-FairnessVQA run:

```bash
python analysis/fairness_posthoc.py \
  --log-path logs/qwen3vl8b-vllm-mimic/<run_log>.json \
  --meta-q-path data/mimic/medrax_input_all_2000.jsonl \
  --meta-case-path data/mimic/mimic_sample_400.csv \
  --out-dir logs/mimic/analysis/qwen3vl8b-vllm/fairness_posthoc \
  --enable-llm-judge
```

`--enable-llm-judge` uses an OpenAI-compatible judge API. By default it expects:

```bash
export DEEPSEEK_API_KEY="..."
```

You can change the judge backend:

```bash
python analysis/fairness_posthoc.py \
  ... \
  --enable-llm-judge \
  --judge-model deepseek-chat \
  --judge-base-url https://api.deepseek.com/v1 \
  --judge-api-key-env DEEPSEEK_API_KEY
```

Core paper outputs include:

- `group_performance.csv`
- `lens_inherited_tool_bias.csv`
- `lens_agentic_transition_rates_by_group.csv`
- `lens_agentic_transition_divergence.csv`
- `lens_agentic_conditional_tool_utility.csv`
- `lens_agentic_conditional_tool_utility_gap.csv`
- `lens_llm_reasoning_bias.csv`
- `summary_report.md`

## Paper Driver Models

The paper evaluates five driver LLMs:

- LLaMA3.1-8B
- Ministral-3-8B
- Qwen3VL-8B
- Qwen3-8B
- Gemini3-Flash

Use `--model`, `OPENAI_BASE_URL`, and the native Gemini flags in [launch_over_chexbench.py](launch_over_chexbench.py) to point the agent at the corresponding backend.

## Notes On Scope

The public release intentionally avoids shipping unpublished result tables, private logs, model weights, and full MIMIC-CXR data. Users must obtain source datasets and model weights from their official providers and comply with their licenses and access terms.

Some scripts contain optional exploratory analyses retained for transparency and future work. The README commands above are the paper reproduction path.

## Acknowledgements

This repository builds on MedRAX, which is released under the Apache-2.0 license. See [NOTICE](NOTICE) for attribution.

## Citation

```bibtex
@article{xu2026ducx,
  title={DUCX: Decomposing Unfairness in Tool-Using Chest X-ray Agents},
  author={Xu, Zikang and Jin, Ruinan and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2603.00777},
  year={2026}
}
```
