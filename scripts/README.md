# Experiment entry points

`run_paper_experiment.py` is the canonical paper entry point. It creates an
isolated timestamped run directory and writes a run manifest.

`download_data.py`, `prepare_data.py`, `generate_tasks.py`, and
`vendor_baselines.py` prepare data and provenance. `run_experiment.py` and
`run_agent_tasks.py` are retained as deprecated component-level debugging
commands. StatQA generation and evaluation use `run_statqa.py` and
`evaluate_statqa_ranker.py` under the canonical entry point. No other script
defines a paper protocol.
