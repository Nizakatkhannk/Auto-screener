"""
Web server for the Systematic Review Auto-Screener.

Pure Python standard library for HTTP + sqlite3 for the audit log;
NumPy for the model (see screener.py). Run it with:

    python app.py

then open http://localhost:8000
"""
import csv
import io
import json
import os
import sqlite3
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np

from screener import (Tfidf, LogisticRegression, keyword_score,
                      run_simulation, top_terms)

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
DB_PATH = os.path.join(BASE, "screener.db")

# The frontend is embedded here (rather than a standalone .html file) so
# the whole application ships as Python source; style.css and app.js live
# in static/ and are served alongside it.
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auto-Screener — Systematic Review Screening</title>
<link rel="stylesheet" href="style.css">
</head>
<body>

<header class="hero">
  <div class="wrap">
    <div class="logo">AUTO<span>SCREENER</span></div>
    <h1>Systematic review screening, with a human in the loop</h1>
    <p class="sub" id="topicLine"></p>
    <p class="sub small">The model ranks papers by estimated relevance and learns from your
      decisions. <strong>It never excludes a paper by itself</strong> — every final decision is
      yours, and every decision is logged with a reason.</p>
  </div>
</header>

<nav class="tabs wrap">
  <button class="tab active" data-tab="screen">1 · Screening</button>
  <button class="tab" data-tab="prisma">2 · PRISMA &amp; Export</button>
  <button class="tab" data-tab="eval">3 · Evaluation</button>
  <button class="tab" data-tab="setup">4 · Setup &amp; Import</button>
</nav>

<main class="wrap">

  <!-- ================= SCREENING ================= -->
  <section id="tab-screen" class="tabpanel">
    <div class="grid">
      <aside class="card side">
        <h3>Progress</h3>
        <div id="progress"></div>
        <h3>Model</h3>
        <div id="modelBox"></div>
        <h3>Up next in the queue</h3>
        <div id="queuePreview" class="queue"></div>
      </aside>

      <div class="card">
        <div id="cardArea"></div>
      </div>
    </div>
  </section>

  <!-- ================= PRISMA ================= -->
  <section id="tab-prisma" class="tabpanel hidden">
    <div class="card">
      <h2>PRISMA-aligned flow counts</h2>
      <p class="hint">These are the counts a PRISMA 2020 flow diagram needs for the
        study-selection stage. They update live as you screen.</p>
      <div class="flow" id="flow"></div>
      <h3>Exclusion reasons</h3>
      <div id="reasons"></div>
      <div class="actions">
        <a class="btn" href="/api/export.csv">Download decisions (CSV)</a>
        <a class="btn ghost" href="/api/summary.json" target="_blank">Download summary (JSON)</a>
        <button class="btn danger" id="resetBtn">Reset session (reload demo data)</button>
      </div>
      <p class="muted">Decisions are also written to the local SQLite audit log
        (<code>screener.db</code>) with paper id, label, reason, timestamp and model version.</p>
    </div>
  </section>

  <!-- ================= EVALUATION ================= -->
  <section id="tab-eval" class="tabpanel hidden">
    <div class="card">
      <h2>Offline evaluation — active-learning simulation</h2>
      <p class="hint">Runs the four ranking strategies on the demo dataset with its known
        labels hidden from the learner (seed: 3 relevant + 3 irrelevant papers, batches of 5).
        This is the experiment behind RQ1–RQ2 of the proposal: does active learning find the
        relevant papers earlier, and how much work does it save?</p>
      <button class="btn" id="simBtn">Run simulation</button>
      <span id="simStatus" class="muted"></span>
      <div id="simResults"></div>
    </div>
  </section>

  <!-- ================= SETUP ================= -->
  <section id="tab-setup" class="tabpanel hidden">
    <div class="card">
      <h2>Review setup</h2>
      <h3>Inclusion criteria</h3>
      <p id="critIn" class="crit"></p>
      <h3>Exclusion criteria</h3>
      <p id="critEx" class="crit"></p>
      <h3>Import your own records (CSV)</h3>
      <p class="hint">CSV columns: <code>title, abstract, year</code>. An optional
        <code>truth</code> column (1 = include, 0 = exclude) enables the simulation;
        without it the screening workflow still works, but evaluation is disabled.
        Importing replaces the current dataset.</p>
      <input type="file" id="csvFile" accept=".csv">
      <button class="btn" id="importBtn">Import CSV</button>
      <span id="importStatus" class="muted"></span>
      <h3>Demo dataset</h3>
      <p class="hint">36 papers on machine learning for early detection of type 2 diabetes
        in adults (10 relevant, 26 near-misses), with ground-truth labels used only by the
        simulation — never by the screening model.</p>
      <button class="btn ghost" id="demoBtn">Reload demo dataset</button>
    </div>
  </section>

  <footer>
    Systematic Review Auto-Screener · Semester project prototype ·
    TF-IDF + class-weighted logistic regression · Relevance + uncertainty active learning ·
    No record is ever excluded automatically.
  </footer>
</main>

<script src="app.js"></script>
</body>
</html>
"""

REVIEW_TOPIC = ("Machine learning for early detection of type 2 diabetes "
                "in adults")

DEFAULT_CRITERIA = {
    "inclusion": ("Adults (18+ years). Machine learning model for early "
                  "detection or prediction of type 2 diabetes. Human study. "
                  "English language. Published 2015 or later."),
    "exclusion": ("Type 1 or gestational diabetes. Children or adolescents. "
                  "Animal studies. Diabetes management, treatment or "
                  "complication prediction (not detection). Non-ML risk "
                  "scores. Reviews and editorials."),
}

INCLUDE_TERMS = ["type 2 diabetes", "machine learning", "early detection",
                 "prediction", "predict", "risk", "adults", "screening",
                 "onset", "hba1c", "undiagnosed"]
EXCLUDE_TERMS = ["type 1 diabetes", "gestational", "children", "adolescent",
                 "mouse", "animal", "management", "complication",
                 "readmission", "review"]

BATCH = 5          # labels before the model retrains
MIX_TOP = 3        # in each new queue head: top-relevance picks
EXCLUSION_REASONS = ["Wrong population", "Wrong intervention",
                     "Wrong study design", "Wrong outcome",
                     "Not English", "Other"]


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.load_demo()

    # ------------------------------------------------------------ data
    def load_demo(self):
        self.papers = []
        with open(os.path.join(BASE, "data", "demo_review.csv"),
                  encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.papers.append({
                    "id": int(row["id"]), "title": row["title"],
                    "abstract": row["abstract"], "year": int(row["year"]),
                    "truth": int(row["truth"]),
                })
        self._reset_model_state()
        self.criteria = dict(DEFAULT_CRITERIA)
        self.review_topic = REVIEW_TOPIC

    def import_rows(self, rows):
        self.papers = []
        for i, r in enumerate(rows, start=1):
            if not r.get("title") or not r.get("abstract"):
                raise ValueError(f"Row {i}: title and abstract are required.")
            self.papers.append({
                "id": i, "title": str(r["title"]).strip(),
                "abstract": str(r["abstract"]).strip(),
                "year": int(r.get("year") or 0), "truth": None,
            })
        self._reset_model_state()

    def _reset_model_state(self):
        self.decisions = {}          # paper_id -> {label, reason, ts, version}
        self.unsure = set()
        self.model = None
        self.vectorizer = None
        self.X = None
        self.model_version = 0
        self.labels_since_train = 0
        self.train_log = []          # [{version, n_labels, ts}]
        self._fit_tfidf()

    def _fit_tfidf(self):
        texts = [p["title"] + ". " + p["abstract"] for p in self.papers]
        self.texts = texts
        self.vectorizer = Tfidf().fit(texts)
        self.X = self.vectorizer.transform(texts)
        self.kw_scores = [keyword_score(t, INCLUDE_TERMS, EXCLUDE_TERMS)
                          for t in texts]

    # ------------------------------------------------------------ ranking
    def _scores(self):
        """Relevance score for every paper (model prob, else keyword)."""
        if self.model is not None:
            return self.model.predict_proba(self.X)
        kw = np.array(self.kw_scores, dtype=float)
        lo, hi = kw.min(), kw.max()
        return (kw - lo) / (hi - lo + 1e-9)

    def queue(self):
        """Current screening order for undecided papers.

        Keyword order until the first model exists; afterwards the head of
        the queue mixes the most relevant records with the most uncertain
        ones (active learning), the rest ordered by relevance.
        """
        undecided = [i for i, p in enumerate(self.papers)
                     if p["id"] not in self.decisions and p["id"] not in self.unsure]
        scores = self._scores()
        if self.model is None:
            return sorted(undecided, key=lambda i: -self.kw_scores[i])
        by_rel = sorted(undecided, key=lambda i: -scores[i])
        unc = sorted(undecided, key=lambda i: abs(scores[i] - 0.5))
        head, seen = [], set()
        for i in by_rel[:MIX_TOP] + unc[: BATCH - MIX_TOP]:
            if i not in seen:
                head.append(i)
                seen.add(i)
        return head + [i for i in by_rel if i not in seen]

    def _retrain_if_due(self):
        labels = [d["label"] for d in self.decisions.values()]
        if self.labels_since_train >= BATCH and len(set(labels)) == 2:
            idx = [i for i, p in enumerate(self.papers)
                   if p["id"] in self.decisions]
            y = np.array([1 if self.decisions[self.papers[i]["id"]]["label"] == "include" else 0
                          for i in idx], dtype=float)
            self.model = LogisticRegression().fit(self.X[idx], y)
            self.model_version += 1
            self.labels_since_train = 0
            self.train_log.append({
                "version": self.model_version,
                "n_labels": len(idx),
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            return True
        return False

    # ------------------------------------------------------------ actions
    def decide(self, paper_id, label, reason=None):
        with self.lock:
            if label == "unsure":
                self.unsure.add(paper_id)
                return {"retrained": False}
            self.decisions[paper_id] = {
                "label": label, "reason": reason or "",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": self.model_version,
            }
            self.unsure.discard(paper_id)
            self.labels_since_train += 1
            retrained = self._retrain_if_due()
            db_log_decision(paper_id, label, reason or "",
                            self.decisions[paper_id]["ts"], self.model_version)
            return {"retrained": retrained}

    def counts(self):
        labels = [d["label"] for d in self.decisions.values()]
        reasons = {}
        for d in self.decisions.values():
            if d["label"] == "exclude" and d["reason"]:
                reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        return {
            "identified": len(self.papers),
            "screened": len(self.decisions),
            "included": labels.count("include"),
            "excluded": labels.count("exclude"),
            "unsure": len(self.unsure),
            "remaining": len(self.papers) - len(self.decisions) - len(self.unsure),
            "exclusion_reasons": reasons,
            "model_version": self.model_version,
            "labels_until_retrain": max(0, BATCH - self.labels_since_train)
            if self.model is not None or len(set(labels)) == 2 else None,
        }

    def state(self):
        q = self.queue()
        scores = self._scores()
        cards = []
        for i in q[:1]:
            p = self.papers[i]
            cards.append({
                "id": p["id"], "title": p["title"], "abstract": p["abstract"],
                "year": p["year"],
                "score": round(float(scores[i]), 3),
                "top_terms": top_terms(self.vectorizer, self.model, self.X, i),
                "keyword_hits": [t for t in INCLUDE_TERMS if t in self.texts[i].lower()],
            })
        return {
            "review_topic": self.review_topic,
            "criteria": self.criteria,
            "counts": self.counts(),
            "next_card": cards[0] if cards else None,
            "queue_preview": [
                {"id": self.papers[i]["id"], "title": self.papers[i]["title"],
                 "score": round(float(scores[i]), 3)} for i in q[1:6]
            ],
            "train_log": self.train_log,
            "exclusion_reasons": EXCLUSION_REASONS,
            "model_active": self.model is not None,
            "total": len(self.papers),
        }

    def simulate(self):
        truth = [p["truth"] for p in self.papers]
        if any(t is None for t in truth):
            return {"error": "Simulation needs ground-truth labels; the demo dataset has them, an imported file does not."}
        seed_idx = ([i for i, p in enumerate(self.papers) if p["truth"] == 1][:3]
                    + [i for i, p in enumerate(self.papers) if p["truth"] == 0][:3])
        return run_simulation(self.texts, truth, INCLUDE_TERMS, EXCLUDE_TERMS,
                              seed_idx, batch=BATCH)


SESSION = Session()


# --------------------------------------------------------------------------
# SQLite audit log
# --------------------------------------------------------------------------

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS decisions (
        paper_id INTEGER, label TEXT, reason TEXT,
        ts TEXT, model_version INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS scores (
        paper_id INTEGER, score REAL, model_version INTEGER, ts TEXT)""")
    return conn


def db_log_decision(paper_id, label, reason, ts, version):
    try:
        c = db()
        c.execute("INSERT INTO decisions VALUES (?,?,?,?,?)",
                  (paper_id, label, reason, ts, version))
        c.commit()
        c.close()
    except Exception:
        pass


def json_response(handler, payload, status=200):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            return json_response(self, SESSION.state())

        if path == "/api/simulate":
            return json_response(self, SESSION.simulate())

        if path == "/api/export.csv":
            out = io.StringIO()
            w = csv.writer(out)
            w.writerow(["id", "title", "year", "decision", "reason",
                        "model_version", "timestamp"])
            scores = SESSION._scores()
            for i, p in enumerate(SESSION.papers):
                d = SESSION.decisions.get(p["id"])
                w.writerow([p["id"], p["title"], p["year"],
                            d["label"] if d else "undecided",
                            d["reason"] if d else "",
                            d["version"] if d else "",
                            d["ts"] if d else ""])
            body = out.getvalue().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition",
                             "attachment; filename=screening_decisions.csv")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/summary.json":
            return json_response(self, {
                "review_topic": SESSION.review_topic,
                "criteria": SESSION.criteria,
                "counts": SESSION.counts(),
                "train_log": SESSION.train_log,
                "model": "TF-IDF (word+bigram) + class-weighted logistic "
                         "regression; relevance + uncertainty active learning",
                "note": "PRISMA-aligned counts only. The model never "
                        "excludes a record; every decision is the reviewer's.",
            })

        if path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.path = path
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return json_response(self, {"error": "bad request"}, 400)

        if parsed.path == "/api/decision":
            try:
                pid = int(payload["paper_id"])
                label = payload["label"]
                if label not in ("include", "exclude", "unsure"):
                    raise ValueError
                if label == "exclude" and not payload.get("reason"):
                    return json_response(self, {"error": "Please pick an exclusion reason."}, 400)
            except (KeyError, ValueError):
                return json_response(self, {"error": "bad decision payload"}, 400)
            res = SESSION.decide(pid, label, payload.get("reason"))
            return json_response(self, {**res, **SESSION.state()})

        if parsed.path == "/api/import":
            try:
                SESSION.import_rows(payload.get("rows", []))
            except ValueError as e:
                return json_response(self, {"error": str(e)}, 400)
            return json_response(self, SESSION.state())

        if parsed.path == "/api/reset":
            SESSION.load_demo()
            return json_response(self, SESSION.state())

        return json_response(self, {"error": "not found"}, 404)

    def translate_path(self, path):
        return os.path.join(STATIC, path.lstrip("/"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Auto-Screener running at  http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
