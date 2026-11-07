# Brief: topic search around predicting LLM item-level performance (v0, 2026-11-07)

Written for a smart outsider. Everything here is public information or our own unpublished idea; no competition data and no unpublished results are included.

## Where this comes from

- **Predictive AI Evaluation Competition (PAIEC) at NeurIPS 2026**, run by Stanford AIMS Lab. Public pages: https://aims.stanford.edu/competition , https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PAIEC , Codabench https://www.codabench.org/competitions/17828/ .
- Task: predict the probability that a *subject* (an AI system = model + harness + harness version + reasoning effort, with provider and release date) answers a given benchmark *item* correctly. Item fields include `item_content` (text), `item_features`, `benchmark_id`. Test benchmarks are disjoint from training benchmarks.
- Test-time adaptation: for each subject–benchmark pair, the predictor is scored after 0, 1, 3, 7, 15, 31 revealed labels (optionally chosen by an acquisition function). Metric: Brier ALC = 0.1·B0 + 0.2·B1 + 0.2·B3 + 0.2·B7 + 0.2·B15 + 0.1·B31 (lower is better).
- Training data: public Hugging Face dataset `aims-foundations/measurement-db` (a large subject × item response database) plus sources on the organisers' benchmark inventory.
- The competition is a source of data and a problem statement, not the paper. The user wants a research topic that stands on its own.

## Who we are and the targets

- One author: an MCS student at UIUC (first author is a student → eligible for student research workshops).
- Targets, in order of realism:
  1. **EACL 2027 Student Research Workshop**, direct submission deadline 2026-12-15 (AoE). Short paper 4 pages, long 8 pages; archival or non-archival; ARR anonymity policy. Call: https://2027.eacl.org/calls/srw/
  2. An ACL-family main conference via ACL Rolling Review (cycle and deadline not yet checked).
  3. The PAIEC technical report itself (NeurIPS template), due 2026-11-07 12:00 UTC.
- The topic must be recognisably **NLP** for an ACL venue: a reviewer must not be able to say "this is generic matrix completion; replace text with images and nothing changes".

## Our current method idea (not implemented)

A prior from an LLM forecaster that reads the item text and the subject description, combined with an IRT / logistic prior fitted on measurement-db, then a one-dimensional Bayesian shift per subject–benchmark pair updated from the revealed labels.

## Candidate directions we already see (to be challenged, extended or killed)

1. **Text vs. responses for cold-start:** when no labels on a new benchmark are available, how much does reading the item text (via an LLM or encoder) beat response-only IRT, and at what label budget does response-only IRT overtake it?
2. **What in the item text transfers:** which linguistic or structural features of an item predict difficulty across benchmarks, and which are benchmark-specific artefacts?
3. **Subject description as a predictor:** can the system description (model, harness, reasoning effort, release date) predict ability on a new benchmark — a text-to-ability problem?
4. **Active label acquisition for evaluation:** which items should be revealed first to estimate a new system's accuracy on a new benchmark with the fewest labels?
5. **LLM self-forecasting:** can an LLM predict whether it (or another system) will answer an item correctly, and is it better calibrated than an external assessor?

## Prior work we know of (unverified, from memory; must be read before use)

IRT for NLP evaluation (Lalor et al., EMNLP 2016; Rodriguez et al., ACL 2021; Vania et al., ACL 2021); efficient evaluation (tinyBenchmarks, Polo et al., 2024; metabench); observational scaling laws (Ruan et al., 2024); item difficulty prediction from text (BEA 2024 shared task); "assessor" models that predict a system's success on an instance.
