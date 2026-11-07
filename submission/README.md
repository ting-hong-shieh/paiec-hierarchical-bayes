# PAIEC submission: IRT subject prior + hierarchical Bayesian update

`model.py` implements `predict(input, labeled) -> float` with the Python standard library only.
`prior.json` holds the offline fit from measurement-db (subject abilities and the
hyperparameters of the test-time model). No `labeling.py` is included, so the platform's
default random acquisition is used. No API keys, network calls or Hugging Face models.

Model, for a held-out benchmark b: logit P(y=1) = c_b + lam*theta_s + u_{s,b} + v_i, with
Gaussian priors on c_b, u_{s,b}, v_i. At each budget the MAP of (c_b, u, v) is computed from
all revealed labels on the same `benchmark_id` (any subject), and the prediction integrates
over the Laplace variance (probit approximation).

Rebuild `prior.json` and rerun the local evaluation: see the repository README.
