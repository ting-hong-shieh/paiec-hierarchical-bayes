# PAIEC 2026: hierarchical Bayesian predictor for benchmark-level cold starts

Entry to the [Predictive AI Evaluation Competition](https://aims.stanford.edu/competition) at NeurIPS 2026
by Elias T.-H. Shieh (University of Illinois Urbana-Champaign). Technical report: `paper/main.pdf`.

The predictor (`submission/model.py`, Python standard library only, no network calls) models
logit P(correct) = c_b + lam*theta_s + u_{s,b} + v_i with Gaussian priors, where theta_s is a
subject ability fitted offline on measurement-db. At each label budget it pools every revealed
label on the same anonymous benchmark, from any subject, computes a MAP fit and returns the
posterior-mean probability under a Laplace (probit) approximation.

## Layout

```text
submission/   model.py, prior.json, README.md  -> zip these three files for Codabench
tools/        download, aggregation, prior fit, local replica of the evaluation, analysis, stress test
results/      local replica outputs (JSON) used in the report
paper/        LaTeX source, generated tables, figure scripts
```

## Rerun

Python 3.10+. Data access to `aims-foundations/measurement-db` on Hugging Face requires accepting its
gate; set `HF_TOKEN`.

```bash
python tools/download_measurement_db.py /path/to/cache          # benchmarks/subjects/responses tables, revision 54074453
python tools/build_tables.py --cache /path/to/cache --out data/
python tools/fit_prior.py --data data/ --out submission/prior.json
python tools/simulate.py --data data/ --out results/sim_per_pair.json
python tools/simulate.py --data data/ --out results/sim_shared.json --split shared
python tools/simulate.py --data data/ --out results/sim_per_pair_seed1.json --seed 1
python tools/simulate.py --data data/ --out results/sens_su_2.0.json --only-ours --scale-su 2.0   # etc.
python tools/analyze.py                                          # tables + figure for the paper
python tools/stress_test.py                                      # load / edge-case test of model.py
(cd submission && zip -X ../dist/submission.zip model.py prior.json README.md)
```

The tools need `pandas`, `pyarrow`, `numpy`, `huggingface_hub`, `requests` and `matplotlib`;
`submission/model.py` needs nothing beyond the standard library. Validate the ZIP with
`check_submission_zip.py` from the [official baseline repository](https://github.com/aims-foundations/paiec_baseline).

## Data and licences

Code: MIT (see `LICENSE`). `submission/prior.json` is derived from the AI Measurement Data Bank
(curation licensed CC BY-SA 4.0; individual benchmarks keep their upstream licences). No raw data are
redistributed here.

## Disclosure

Code and the first draft of the report were written with an LLM coding assistant (Claude Code)
under the author's direction.
