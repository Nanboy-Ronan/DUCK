# Legacy Experiment Utilities

This directory contains auxiliary benchmarking scripts inherited from the upstream MedRAX and ChestAgentBench codebase. They are kept for transparency, but they are **not** the default reproduction path for the DUCX paper.

For the paper release, use the commands in the repository-level [README.md](../README.md):

- run agent trajectories with [launch_over_chexbench.py](../launch_over_chexbench.py);
- compute DUCX fairness outputs with [analysis/fairness_posthoc.py](../analysis/fairness_posthoc.py);
- use only the datasets and driver LLMs reported in the paper.

The scripts here may reference baseline models, paths, or datasets that are outside the DUCX paper scope, such as standalone GPT-4o, LLaVA-Med, CheXagent, CheXbench, or MedMAX-style layouts. Treat them as optional developer utilities, not paper claims.
