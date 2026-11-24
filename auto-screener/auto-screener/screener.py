"""
ML core for the Systematic Review Auto-Screener.

Pure NumPy (no scikit-learn needed): TF-IDF vectorisation, class-weighted
logistic regression trained by gradient descent, a keyword baseline, and
relevance + uncertainty active-learning sampling.

The pipeline follows the semester-project proposal:
  1. TF-IDF baseline (word + bigram) + class-weighted logistic regression.
  2. Keyword baseline from the eligibility criteria (interpretable comparator).
  3. Active learning: each batch mixes the highest-relevance records with
     the most uncertain records (closest to the decision boundary).
  4. Evaluation: recall, precision, F1, PR-AUC (average precision) and
     WSS@95 (work saved over random sampling at 95% recall).
"""
import math
import re
from collections import Counter

import numpy as np

STOPWORDS = set("""
a an the and or of to in on for with by at from as is are was were be been being
it its this that these those we our they their he she his her not no yes if then
than into over under between through during before after about above below out up
down off again further once here there when where which who whom how what all any
both each few more most other some such only own same so too very can will just
should now use used using study studies paper papers based method methods model
models result results data analysis analyses research aim aims objective objectives
conclusion conclusions background purpose report reported new also may might could
would among within without across per via et al
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    toks = [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]
    bigrams = [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return toks + bigrams


class Tfidf:
    """Sublinear-TF, smoothed-IDF, L2-normalised TF-IDF (dense NumPy)."""

    def __init__(self, max_features=2500):
        self.max_features = max_features

    def fit(self, docs):
        df = Counter()
        per_doc = []
        for d in docs:
            toks = tokenize(d)
            per_doc.append(Counter(toks))
            for t in set(toks):
                df[t] += 1
        # keep the most frequent terms (document frequency as the ranking)
        vocab = [t for t, _ in df.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(vocab)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in vocab])
        self._per_doc = per_doc
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)), dtype=float)
        for r, d in enumerate(docs):
            for t, c in Counter(tokenize(d)).items():
                j = self.vocab.get(t)
                if j is not None:
                    X[r, j] = (1.0 + math.log(c)) * self.idf[j]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)


class LogisticRegression:
    """Binary logistic regression, full-batch gradient descent, class-weighted."""

    def __init__(self, lr=1.0, l2=0.1, epochs=1000, seed=42):
        self.lr, self.l2, self.epochs = lr, l2, epochs
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        n_pos = int(y.sum())
        n_neg = n - n_pos
        # balanced class weights so the minority (include) class is not ignored
        cw = np.where(y == 1, n / (2.0 * max(n_pos, 1)), n / (2.0 * max(n_neg, 1)))
        for _ in range(self.epochs):
            z = X @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            err = (p - y) * cw
            gw = X.T @ err / n + (self.l2 / n) * w
            gb = float(err.sum()) / n
            w -= self.lr * gw
            b -= self.lr * gb
        self.w, self.b = w, b
        return self

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# --------------------------------------------------------------------------
# Keyword baseline
# --------------------------------------------------------------------------

def keyword_score(text, include_terms, exclude_terms):
    """Simple interpretable comparator: overlap with eligibility terms."""
    low = text.lower()
    hits = sum(1 for t in include_terms if t in low)
    bad = sum(1 for t in exclude_terms if t in low)
    return hits - 0.5 * bad


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def average_precision(order, truth):
    """PR-AUC approximated by average precision over a ranking."""
    n_rel = int(sum(truth))
    if n_rel == 0:
        return 0.0
    hits, s = 0, 0.0
    for k, idx in enumerate(order, start=1):
        if truth[idx] == 1:
            hits += 1
            s += hits / k
    return s / n_rel


def recall_curve(order, truth):
    n_rel = int(sum(truth))
    hits, curve = 0, []
    for k, idx in enumerate(order, start=1):
        if truth[idx] == 1:
            hits += 1
        curve.append(hits / max(n_rel, 1))
    return curve


def wss_at_95(order, truth):
    """Work Saved over Sampling at 95% recall. 0.95 - (screened / total)."""
    n = len(order)
    curve = recall_curve(order, truth)
    for k, r in enumerate(curve, start=1):
        if r >= 0.95:
            return round(0.95 - k / n, 3)
    return round(0.95 - 1.0, 3)


def metrics_for_order(order, truth, threshold_rank=None):
    """Precision / recall / F1 at the top-k cut, plus PR-AUC and WSS@95."""
    n_rel = int(sum(truth))
    k = threshold_rank if threshold_rank is not None else n_rel
    top = order[:k]
    tp = sum(1 for i in top if truth[i] == 1)
    precision = tp / max(len(top), 1)
    recall = tp / max(n_rel, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "pr_auc": round(average_precision(order, truth), 3),
        "wss95": wss_at_95(order, truth),
    }


# --------------------------------------------------------------------------
# Active-learning simulation (labels hidden from the learner, used for eval)
# --------------------------------------------------------------------------

def simulate(X, truth, include_terms, exclude_terms, seed_idx, batch=5,
             mix_top=3, seed=42):
    """Returns orderings for: random, keyword, TF-IDF static, TF-IDF active."""
    n = len(truth)
    rng = np.random.default_rng(seed)

    # --- random control (seeded shuffle) ---
    order_random = list(rng.permutation(n))

    # --- keyword baseline (static ranking) ---
    texts = X_texts  # set by the caller via module global (see run_simulation)
    kw = [keyword_score(t, include_terms, exclude_terms) for t in texts]
    order_keyword = sorted(range(n), key=lambda i: -kw[i])

    # --- TF-IDF static: train once on the seed, rank everything ---
    labeled = list(seed_idx)
    y = np.array([truth[i] for i in labeled], dtype=float)
    model = LogisticRegression().fit(X[labeled], y)
    probs = model.predict_proba(X)
    order_static = sorted(range(n), key=lambda i: -probs[i])

    # --- TF-IDF active learning: seed, then batches of relevance+uncertainty ---
    labeled = list(seed_idx)
    order_active = list(seed_idx)
    remaining = [i for i in range(n) if i not in set(labeled)]
    while remaining:
        y = np.array([truth[i] for i in labeled], dtype=float)
        m = LogisticRegression().fit(X[labeled], y)
        p = m.predict_proba(X[remaining])
        # mix: highest relevance + most uncertain
        rel = sorted(range(len(remaining)), key=lambda j: -p[j])
        unc = sorted(range(len(remaining)), key=lambda j: abs(p[j] - 0.5))
        pick, seen = [], set()
        for j in rel[:mix_top] + unc[: batch - mix_top]:
            if j not in seen:
                pick.append(remaining[j])
                seen.add(j)
        for j in pick:
            order_active.append(j)
            labeled.append(j)
        remaining = [i for i in range(n) if i not in set(labeled)]
    return {
        "random": order_random,
        "keyword": order_keyword,
        "tfidf_static": order_static,
        "tfidf_active": order_active,
    }


X_texts = []  # filled by run_simulation before calling simulate()


def run_simulation(texts, truth, include_terms, exclude_terms, seed_idx, batch=5):
    global X_texts
    X_texts = texts
    vec = Tfidf().fit_transform(texts)
    orders = simulate(vec, truth, include_terms, exclude_terms, seed_idx, batch=batch)
    out = {}
    for name, order in orders.items():
        m = metrics_for_order(order, truth)
        m["recall_curve"] = [round(v, 3) for v in recall_curve(order, truth)]
        out[name] = m
    return out


def top_terms(vectorizer, model, X, doc_idx, topn=3):
    """Influential terms for one paper: TF-IDF weight x model coefficient.

    This is the explainability signal shown on each screening card: the
    terms that pushed the score up (positive) or down (negative).
    """
    if model is None:
        return []
    contrib = X[doc_idx] * model.w
    idx = np.argsort(-np.abs(contrib))[:topn]
    inv = {j: t for t, j in vectorizer.vocab.items()}
    out = []
    for j in idx:
        if abs(contrib[j]) > 1e-9:
            out.append({"term": inv[j].replace("_", " "),
                        "effect": "up" if contrib[j] > 0 else "down"})
    return out
