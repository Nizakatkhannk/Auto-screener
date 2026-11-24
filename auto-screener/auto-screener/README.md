# Systematic Review Auto-Screener

A human-in-the-loop machine-learning system for title-and-abstract
screening — the working implementation of the semester-project proposal
*“Systematic Review Auto-Screener: A Human-in-the-Loop Active-Learning
System for Research Paper Screening.”*

It ranks candidate papers by estimated relevance, learns from the
reviewer's decisions, and produces PRISMA-aligned counts and an auditable
export. The model never excludes a paper by itself.

## Run it

```bash
pip install -r requirements.txt     # only NumPy
python app.py
```

Then open **http://localhost:8000** in your browser.

## What it does (the proposal, implemented)

| Proposal requirement | Where it lives |
|---|---|
| Import bibliographic records (CSV) | Setup tab — import CSV, or the built-in demo dataset |
| Inclusion / exclusion criteria | Setup tab (drives the keyword baseline) |
| TF-IDF baseline + class-weighted logistic regression | `screener.py` (pure NumPy; scikit-learn not required) |
| Active learning: relevance + uncertainty sampling | Queue re-ranks after every 5 labels (`BATCH = 5` in `app.py`) |
| Human control, reasons, audit log | Screening card (Include / Exclude + reason / Unsure); decisions in SQLite (`screener.db`) |
| PRISMA-aligned counts + export | PRISMA tab — flow counts, decisions CSV, summary JSON |
| Evaluation: recall, precision, F1, PR-AUC, WSS@95 | Evaluation tab — offline simulation vs random, keyword and static TF-IDF |

## How a screening session works

1. **Setup** — the demo review is loaded: *“Machine learning for early
   detection of type 2 diabetes in adults”*, 36 papers (10 relevant,
   26 near-misses). Ground-truth labels exist only for the simulation;
   the screening model never sees them.
2. **Screening** — papers arrive one card at a time with a relevance
   score and the terms that pushed the score up or down. Decide
   **Include**, **Exclude** (with a reason) or **Unsure** (re-queued
   later). The queue starts in keyword order; after 5 labels with at
   least one include and one exclude, the TF-IDF model trains and the
   queue re-ranks — the head of each new queue mixes the most relevant
   papers with the most uncertain ones.
3. **PRISMA & Export** — live flow counts (identified → screened →
   included / excluded, with reasons) and one-click CSV + JSON export.
4. **Evaluation** — “Run simulation” compares four ranking strategies
   on the demo dataset (seed: 3 relevant + 3 irrelevant; batches of 5)
   and draws the recall curves. The result is reported honestly: on this
   small curated demo set the keyword baseline is strong (eligibility
   terms line up with the topic), static TF-IDF still saves roughly half
   the work at 95% recall, and active learning trades some early recall
   for uncertainty sampling — its payoff is model improvement during the
   live session. The mixture parameter is a finding to report (RQ2),
   not a fixed constant.

## Files

| File | What it is |
|---|---|
| `app.py` | Web server (Python standard library + sqlite3) and the screening session |
| `screener.py` | TF-IDF vectoriser, logistic regression, keyword baseline, active-learning simulation, metrics |
| `data/demo_review.csv` | Demo dataset: 36 papers with hidden ground-truth labels |
| `static/app.js`, `static/style.css` | Frontend logic and design (the page itself is embedded in `app.py`) |
| `screener.db` | Created on first run — SQLite audit log of every decision |

## Notes for the report

- The proposal's sentence-embedding variant (sentence-transformers) is a
  documented extension: `screener.py` isolates representation from
  ranking, so embeddings can replace the TF-IDF matrix without changing
  the active-learning loop. The shipped prototype uses the TF-IDF active
  learner, which the proposal explicitly accepts as satisfying the core
  research question when transformer inference is impractical.
- All results are reproducible: fixed random seeds, a logged model
  version per decision, and the full decision history in the export.
