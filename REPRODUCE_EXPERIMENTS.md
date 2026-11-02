# Reproducing TrustStatAgent experiments

Raw data, API keys, model outputs, and generated tables stay outside Git. Official download links are listed in `DATA_SOURCES.txt`.

## Environment

```bash
conda create -n truststatagent python=3.11 -y
conda activate truststatagent
pip install -e .
```

Create a local ignored `.env` containing the required API variables. Never commit this file.

## Data

```bash
python scripts/download_data.py --config configs/default.yaml
python scripts/prepare_data.py --config configs/default.yaml
```

Preparation writes separate train, calibration, validation, and test splits under `data/processed/`.

## Numeric experiments

Parameters are selected on validation only. Test labels are used only for final metrics.

```bash
PYTHONPATH=src python scripts/run_numeric_experiments.py --config configs/default.yaml --reuse-agent-traces
```

For classification, `coverage` is unconditional conformal prediction-set coverage against the true held-out label. `released_correctness` is correctness conditional on automatic release. They are reported separately because a non-singleton set can contain the true label while still being sent to review.

## StatQA mini

The formal split is:

```text
scorer_calibration -> conformal_calibration -> validation -> test
```

Generate candidates:

```bash
for method in truststatagent react without_explicit_planning; do
  for split in scorer_calibration conformal_calibration validation test; do
    PYTHONPATH=src python scripts/run_statqa.py --config configs/default.yaml --dataset mini --method "$method" --mode api --tag opt_mini_full --workers 8 --consistency-k 1 --skip-scorer --split "$split"
  done
done
```

Evaluate the local fixed ranker and conformal gate:

```bash
PYTHONPATH=src python scripts/evaluate_statqa_ranker.py --dataset mini --tag opt_mini_full --target-release 0.50 --max-error-release 0.05 --output results/metrics/statqa_mini_ranker_opt_full.jsonl
```

The ranker is fitted only on scorer-calibration. The conformal quantile is fitted only on conformal-calibration. One global operating point is selected on validation. Test is used only for final evaluation.

## Export tables

```bash
PYTHONPATH=src python scripts/export_paper_tables.py
```

The workbook is written to `results/tables/TrustStatAgent_ESWA_tables.xlsx` and contains task performance, baseline reliability, planning ablation, module ablation, and StatQA category results.

## Multi-seed evaluation

The frozen StatQA candidates published under `results/raw/` can be rescored offline under repeated split assignments without further API calls:

```bash
PYTHONPATH=src python scripts/run_statqa_multiseed.py \
  --verbalized results/raw/statqa_mini_verbalized_confidence_opt_mini_full.jsonl \
  --output results/metrics/statqa_mini_multiseed.jsonl
```

Seven seeds are evaluated. Seed 42 reuses the official stratified split released with the benchmark and reproduces the single-configuration report; seeds 43 to 48 reshuffle the assignment of tasks to the scorer-calibration, conformal-calibration, validation, and test splits with the same per-category ratios. The output records, per variant and seed, the release statistics, the raw confusion counts (TP, FN, TN, FP), and the validation AUC of the trained correctness score and of the verbalized-confidence signal.

The numeric benchmarks are evaluated over three seeds by `run_numeric_experiments.py` (see `experiments.seeds` in `configs/default.yaml`). Pipelines whose metrics are deterministic given the data report zero variance across seeds.

## Verbalized confidence baseline

The self-reported-confidence gating baseline elicits one probability of correctness per frozen candidate:

```bash
PYTHONPATH=src python scripts/run_statqa_verbalized_confidence.py --config configs/default.yaml --dataset mini --tag opt_mini_full --workers 8
```

This requires API access and writes `results/raw/statqa_mini_verbalized_confidence_opt_mini_full.jsonl`; the multi-seed script above consumes it.

## Publication checks

```bash
git check-ignore -v .env
git status --short
```

Do not commit `.env`, raw or processed data, generated model outputs other than the frozen artifacts listed in `.gitignore`, or internal paper drafts. Each numeric run writes a manifest containing configuration, seeds, split sizes, selected parameters, model versions, and a dataset revision hash.
