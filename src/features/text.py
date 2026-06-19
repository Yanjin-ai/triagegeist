"""Phase 3 v1 — TF-IDF text branch on chief_complaint_raw.

word (1-2) + char (3-5) TF-IDF → multinomial LogisticRegression → 5-class proba.
Vocabulary is frozen at fit time (fit on train only) → no train/serve skew, no leakage.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..models.tabular import CLASSES


class TextBranch:
    def __init__(self, seed: int = 42):
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=3, sublinear_tf=True
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
        )
        self.clf = LogisticRegression(
            max_iter=2000, C=3.0, class_weight="balanced", random_state=seed,
        )
        self.fitted_ = False

    @staticmethod
    def _clean(texts) -> list[str]:
        return [str(t).lower().strip() for t in texts]

    def _features(self, texts, fit: bool):
        texts = self._clean(texts)
        if fit:
            w = self.word_vec.fit_transform(texts)
            c = self.char_vec.fit_transform(texts)
        else:
            w = self.word_vec.transform(texts)
            c = self.char_vec.transform(texts)
        return hstack([w, c]).tocsr()

    def fit(self, texts, y) -> "TextBranch":
        X = self._features(texts, fit=True)
        self.clf.fit(X, np.asarray(y).astype(int))
        self.fitted_ = True
        return self

    def predict_proba(self, texts) -> np.ndarray:
        proba = self.clf.predict_proba(self._features(texts, fit=False))
        out = np.zeros((proba.shape[0], len(CLASSES)))
        for j, cls in enumerate(self.clf.classes_):
            out[:, np.where(CLASSES == cls)[0][0]] = proba[:, j]
        return out

    @property
    def vocab_size(self) -> int:
        return len(self.word_vec.vocabulary_) + len(self.char_vec.vocabulary_)
