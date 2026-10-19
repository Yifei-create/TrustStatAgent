# TrustStatAgent

TrustStatAgent is a reproducible statistical-agent benchmark. The LLM plans the
analysis and calls a fixed statistical tool registry. Python performs all data
access, model fitting, prediction, conformal calibration, validation, gating,
and evaluation.

## Reproducible workflow

```bash
python -m pip install -e .
python scripts/download_data.py --all
python scripts/prepare_data.py --config configs/default.yaml
python scripts/generate_tasks.py
# Canonical paper entry point. It runs numeric experiments and the four-way
# StatQA pipeline with independent output tags.
python scripts/run_paper_experiment.py --config configs/default.yaml --mode api --dataset mini
```

For a local smoke run, use the dry-run configuration and skip the numeric LLM
preflight:

```bash
python scripts/run_paper_experiment.py --config configs/debug.yaml --mode dry-run --dataset mini --statqa-limit 20 --skip-numeric-agent
```

`run_experiment.py`, `run_agent_tasks.py`, and `run_statqa.py` remain available
for component-level debugging. They are not the formal paper entry point.

Set `OPENAI_API_KEY` only when API-backed runs are intended. The key is never
stored in this repository. The debug configuration limits task counts; the
default configuration points at the complete official StatQA files.


The final paper runs must record `configs/default.yaml`, the data manifest,
source repository revisions, model identifier, and result files. Test labels
are loaded only by the evaluator and are never returned by the tool registry.

California Housing is split deterministically into train/calibration/validation/test
(50/15/15/20). Adult uses the official test file as the final test set and
splits the official training file into fit/calibration/validation (60/20/20).
Carbon Monitor uses chronological 50/15/15/20 splits. Conventional gate
thresholds are selected on validation only; conformal quantities use calibration
only.

## Data and baseline provenance

Official sources:

- StatQA: https://github.com/HKUSTDial/StatQA
- Zero-shot-CoT: https://github.com/kojima-takeshi188/zero_shot_cot
- ReAct: https://github.com/ysymyth/ReAct
- California Housing: scikit-learn `fetch_california_housing`
- Adult: https://archive.ics.uci.edu/dataset/2/adult
- Carbon Monitor: https://zenodo.org/records/5611986

Baseline source revisions are written to `third_party/provenance.json` by the
vendor script. The statistical tool adapter is local and shared by ReAct and
TrustStatAgent so that the workflow comparison uses identical models and data.

`run_statqa.py` uses the official prompt files and the official columns/methods
selection criterion. Use `--dataset full --mode api` only after the API key is
configured. The debug configuration deliberately limits calls so that API
costs remain small during development.
