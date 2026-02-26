# DUCX: Decomposing Unfairness in Tool-Using Chest X-ray Agents

Anonymous working code release for paper submission.

## 1) Environment Setup

Requirements:
- Python >= 3.10
- CUDA GPU (recommended)
- `git-lfs` (recommended for large assets)

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Optional (for local OpenAI-compatible serving):

```bash
pip install vllm
```

## 2) Data Setup

Create these directories under repo root:

```bash
mkdir -p data/chestagentbench
mkdir -p data/mimic
mkdir -p figures
mkdir -p logs
mkdir -p model-weights temp
```

Expected files for ChestAgentBench evaluation:
- `data/chestagentbench/metadata.jsonl`
- `data/eurorad_metadata.json`
- extracted images under `figures/` (paths in metadata should resolve relative to this folder)

Optional MIMIC evaluation files:
- `data/mimic/medrax_input_all_2000.jsonl`

## 3) Start vLLM (OpenAI-Compatible)

Example with Qwen3-VL-8B:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Set API variables for the agent:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
export OPENAI_MODEL="qwen3-vl-8b"
```

## 4) Launch Agent Runs

### ChestAgentBench

```bash
python launch_over_chexbench.py \
  --model "${OPENAI_MODEL}" \
  --model-dir ./model-weights \
  --temp-dir ./temp \
  --data-file data/chestagentbench/metadata.jsonl \
  --device cuda \
  --log-prefix qwen3vl8b-vllm
```

### MIMIC-CXR-QA (optional)

```bash
python launch_over_chexbench.py \
  --model "${OPENAI_MODEL}" \
  --model-dir ./model-weights \
  --temp-dir ./temp \
  --data-file data/mimic/medrax_input_all_2000.jsonl \
  --device cuda \
  --log-prefix qwen3vl8b-vllm-mimic
```

Logs are written under `./logs/<log-prefix>/`.

## 5) Launch Analysis

### A. Fairness decomposition (single run)

```bash
python analysis/fairness_posthoc.py \
  --log-path ./logs/qwen3vl8b-vllm/<run_log>.json \
  --meta-q-path ./data/chestagentbench/metadata.jsonl \
  --meta-case-path ./data/eurorad_metadata.json \
  --out-dir ./logs/chexagentbench/analysis/qwen3vl8b-vllm/fairness_posthoc
```

### B. Fairness decomposition in batch

```bash
python analysis/fairness_posthoc.py \
  --input-root ./logs/chexagentbench \
  --out-root ./logs/chexagentbench/analysis \
  --meta-q-path ./data/chestagentbench/metadata.jsonl \
  --meta-case-path ./data/eurorad_metadata.json
```

### C. Gap decomposition

```bash
python analysis/decompose_bias_from_fairness.py \
  --analysis-root ./logs/chexagentbench/analysis \
  --out-dir ./logs/chexagentbench/analysis/decomposition
```

## Acknowledgement
This repo is developed based on MedRAX.