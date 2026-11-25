"""Baseline components, each producing calibrated 18-class probabilities via grouped out-of-fold prediction.
All vectorizers/scalers/calibrators are fit inside the training fold only."""

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import SEED
from .style import style_matrix
from .text import normalize


def _softmax(z):
    if z.ndim == 1:  # binary decision values -> two-column logits
        z = np.stack([-z / 2, z / 2], 1)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class TfidfLinear:
    """Char(2-5, raw text incl. typography) + word(1-2, normalized) TF-IDF -> LinearSVC; probabilities via temperature-scaled
    decision values (temperature fit on an inner grouped split of the training fold)."""

    def __init__(self, C=0.5, max_char=300_000, max_word=150_000):
        self.C, self.max_char, self.max_word = C, max_char, max_word

    def _feats(self, texts, fit=False):
        norm = [normalize(t) for t in texts]
        if fit:
            self.cv = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=2,
                max_features=self.max_char,
                lowercase=False,
            )
            self.wv = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
                max_features=self.max_word,
                token_pattern=r"(?u)\b\w+\b|[^\w\s]",
            )
            return hstack([self.cv.fit_transform(texts), self.wv.fit_transform(norm)]).tocsr()
        return hstack([self.cv.transform(texts), self.wv.transform(norm)]).tocsr()

    def fit(self, texts, y, groups):
        from sklearn.model_selection import GroupShuffleSplit

        tr, ca = next(
            GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(texts, y, groups)
        )
        texts = np.asarray(texts, dtype=object)
        clf = LinearSVC(C=self.C, class_weight="balanced").fit(
            self._feats(texts[tr], fit=True), y[tr]
        )
        d = clf.decision_function(self._feats(texts[ca]))
        self.T = _fit_temperature(d, y[ca])
        self.clf = LinearSVC(C=self.C, class_weight="balanced").fit(self._feats(texts, fit=True), y)
        return self

    def predict_proba(self, texts):
        return _softmax(self.clf.decision_function(self._feats(list(texts))) / self.T)


class StyleLR:
    def __init__(self, C=0.3):
        self.C = C

    def fit(self, texts, y, groups=None):
        X = style_matrix(texts)
        self.sc = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(
            self.sc.transform(X), y
        )
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.sc.transform(style_matrix(texts)))


def _fit_temperature(d, y):
    best, bt = 1e9, 1.0
    for T in np.geomspace(0.05, 5, 60):
        p = _softmax(d / T)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best:
            best, bt = nll, T
    return bt


def blend_log(probs_list, weights):
    z = sum(w * np.log(p + 1e-9) for p, w in zip(probs_list, weights))
    return _softmax(z)
