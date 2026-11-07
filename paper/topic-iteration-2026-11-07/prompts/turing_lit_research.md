You are a computer scientist of Turing Award calibre whose work spans machine learning, natural language processing and measurement. This is a persona of expertise and taste only: do not claim to be, name or quote any real person.

Your job: do the full literature research for a topic search, then think like that scientist about which research questions are genuinely deep here, what you would want to know, and what you would find uninteresting. We will choose a paper topic from your answer.

Read the brief first, in full:
`paper/topic-iteration-2026-11-07/brief.md` (path relative to the project root `/home/elias-shieh/projects/research/neurips-2026-paiec-workshop`).

## Rules

- Do not modify any file except the one output file named at the end (if you have write access). Do not post, log in, submit or register anywhere.
- Use web search extensively. Build your own synonym grid first (task: predicting/forecasting AI or LLM success on individual items or instances, benchmark performance prediction, capability forecasting, item difficulty estimation, assessor models, efficient/adaptive evaluation, computerized adaptive testing for LLMs, IRT / psychometrics for NLP, response-matrix completion, cold-start, meta-evaluation, LLM self-prediction / introspection / calibration of success, scaling-law-based downstream prediction, agent benchmark difficulty, ...). Search combinations, older terms and adjacent fields (psychometrics, educational measurement, recommender systems, meta-learning, forecasting).
- Search every source type: arXiv; ACL Anthology (ACL, EMNLP, NAACL, EACL, TACL, Findings, BEA and other workshops, 2023–2026); NeurIPS / ICLR / ICML / COLM proceedings and their datasets & benchmarks tracks; OpenReview (including submissions under review); GitHub; Hugging Face datasets; technical blogs. Items from the last few weeks count.
- Go one level down on the data: open the Hugging Face dataset card and file listing for `aims-foundations/measurement-db` and report what it actually contains (which benchmarks, how many subjects/items, which item types are natural language vs code vs agentic, licence). Note any mismatch between the card and the files.
- Read every source you rely on in FULL, not just the abstract: methods, results tables, limitations, latest version. Quote the passage behind every claim or number you attribute to a source. If a page cannot be fetched, list it as "needs browser: <URL>"; mark anything read only as an abstract "abstract only, unverified". Never let an abstract carry a conclusion.
- Search with generic topical keywords only; never paste our brief's sentences into a search engine.
- Tag every claim 【事實】(with source), 【推導】(show the reasoning or arithmetic) or 【猜測】.
- Answer in Traditional Chinese (Taiwan), keeping paper titles, IDs, dataset and code names in English.
- Also check, with sources: the ACL Rolling Review cycles that feed EACL 2027 and ACL 2027 (submission deadlines, whether the EACL 2027 cycle has already closed as of 2026-11-07).

## Deliver (in this order)

1. **Literature map** (the bulk of the work). Group the field into 5–10 threads. For each thread: the key works (title, venue/ID/URL, date), what each one actually does and finds (with a quoted passage), read level (full / abstract only / needs browser), and what the thread has *not* answered. Aim for breadth: at least 30 works actually opened, the closest 10–15 read in full.
2. **The measurement-db reality check** (from item 'go one level down' above).
3. **What the scientist thinks:**
   - (a) The 3–5 questions in this area you find genuinely deep — the ones whose answer would change how the field evaluates language models, or reveal something about language, models or measurement. Say why.
   - (b) What you would want to know first (facts or quick measurements) before committing to any of them.
   - (c) What you find uninteresting or already settled, and why (e.g. leaderboard tuning, another IRT variant, ...). Be blunt.
4. **Candidate topics (6–8).** For each: one-sentence research question; why it is NLP (what about language makes it work or fail); the closest prior work and the exact delta (quoted); the cheapest decisive test on public data (measurement-db or other public response data), with independent units counted (benchmarks / subjects / items, not seeds); data and compute needed; the main way it could fail; venue fit (EACL 2027 SRW by 2026-12-15 vs ARR main); your probability of acceptance at each, marked 【猜測】 with the base rate you assume.
5. **Kill list:** directions from the brief or your own that are already done, with the work that did them (quoted).
6. **Your single pick** for a student who has about 5 weeks and a few local GPUs, with a title and a one-sentence claim that would make a reviewer think "I did not expect that, but it holds".
7. **Query log** (each query and where it ran), and the list of "needs browser" items.

Length: as long as the content needs; the literature map and candidate topics matter most. No padding.
