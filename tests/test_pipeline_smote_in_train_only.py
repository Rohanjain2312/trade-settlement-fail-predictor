"""SMOTENC sits inside the imblearn pipeline, so it only ever sees training rows: the
training set when fitting, or the training folds inside cross-validation. Validation and test
rows are never resampled. Resampling before the split would leak test-like synthetic points
into training; the last test shows that leak happening the wrong way."""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTENC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import NearestNeighbors

from src.features.definitions import CATEGORICAL, NUMERIC
from src.models import preprocess as P


class SpySMOTENC(SMOTENC):
    """Records every batch of rows it is asked to resample."""

    seen: list = []

    def fit_resample(self, X, y, **kw):
        SpySMOTENC.seen.append(np.asarray(X).copy())
        return super().fit_resample(X, y, **kw)


class SpyModel(LogisticRegression):
    fit_rows: list = []
    predict_rows: list = []

    def fit(self, X, y, sample_weight=None):
        SpyModel.fit_rows.append(len(X))
        return super().fit(X, y, sample_weight)

    def predict_proba(self, X):
        SpyModel.predict_rows.append(len(X))
        return super().predict_proba(X)


def _pipeline():
    pipe = P.build_pipeline(SpyModel(max_iter=500), "smote", seed=0)
    pipe.steps[1] = ("smote", SpySMOTENC(categorical_features=list(range(len(CATEGORICAL))), random_state=0))
    return pipe


def _rows(a):
    return {r.tobytes() for r in np.round(np.asarray(a, dtype=float), 9)}


def test_pipeline_order_is_encode_smote_expand_model():
    assert [name for name, _ in P.build_pipeline(LogisticRegression(), "smote", 0).steps] == [
        "encode", "smote", "expand", "model"]
    assert "smote" not in dict(P.build_pipeline(LogisticRegression(), "none", 0).steps)


def test_smote_resamples_only_the_training_rows(smoke_data):
    splits = P.split_frames(smoke_data.df, smoke_data.cfg)
    X, y = P.xy(splits["train"])
    Xt, yt = P.xy(splits["test"])
    for spy in (SpySMOTENC.seen, SpyModel.fit_rows, SpyModel.predict_rows):
        spy.clear()
    pipe = _pipeline().fit(X, y)
    assert len(SpySMOTENC.seen) == 1 and len(SpySMOTENC.seen[0]) == len(X)
    assert SpyModel.fit_rows[0] > len(X)  # the model saw training rows plus synthetic fails
    pipe.predict_proba(Xt)
    assert len(SpySMOTENC.seen) == 1  # predicting never resamples
    assert SpyModel.predict_rows[-1] == len(Xt)  # test rows pass through unchanged
    encoded_test = pipe.named_steps["encode"].transform(Xt)
    assert not _rows(SpySMOTENC.seen[0]) & _rows(encoded_test)


def test_smote_inside_cross_validation_sees_only_training_folds(smoke_data):
    train = P.split_frames(smoke_data.df, smoke_data.cfg)["train"].sort_values("trade_date")
    X, y = P.xy(train)
    SpySMOTENC.seen.clear()
    folds = list(TimeSeriesSplit(n_splits=3).split(X))
    cross_val_score(_pipeline(), X, y, cv=folds, scoring="average_precision")
    assert len(SpySMOTENC.seen) == 3
    encoder = P.build_pipeline(LogisticRegression(), "none", 0).named_steps["encode"]
    for seen, (tr, va) in zip(SpySMOTENC.seen, folds):
        enc = encoder.fit(X.iloc[tr]).transform
        assert len(seen) == len(tr)
        assert not _rows(seen) & _rows(enc(X.iloc[va]))


def test_resampling_before_the_split_leaks_test_information(smoke_data):
    """The wrong way: SMOTE on all data, then split. Synthetic fails sit right next to test
    fails, so a model 'trained' on them has effectively seen the test set."""
    splits = P.split_frames(smoke_data.df, smoke_data.cfg)
    X_tr, y_tr = P.xy(splits["train"])
    X_te, y_te = P.xy(splits["test"])
    enc = P.build_pipeline(LogisticRegression(), "none", 0).named_steps["encode"].fit(X_tr)
    A_tr, A_te = enc.transform(X_tr), enc.transform(X_te)
    sm = SMOTENC(categorical_features=list(range(len(CATEGORICAL))), random_state=0)
    num = slice(len(CATEGORICAL), len(CATEGORICAL) + len(NUMERIC))
    scale = A_tr[:, num].std(axis=0) + 1e-9
    test_fails = A_te[y_te == 1][:, num] / scale

    def distance_to_nearest_synthetic(synth):
        nn = NearestNeighbors(n_neighbors=1).fit(synth[:, num] / scale)
        return float(np.median(nn.kneighbors(test_fails)[0]))

    # Right way: resample the training rows only.
    right, _ = sm.fit_resample(A_tr, y_tr)
    # Wrong way: resample train and test together, then split.
    both = np.vstack([A_tr, A_te])
    wrong, _ = sm.fit_resample(both, np.concatenate([y_tr, y_te]))
    d_right = distance_to_nearest_synthetic(right[len(A_tr):])
    d_wrong = distance_to_nearest_synthetic(wrong[len(both):])
    # Test fails have synthetic copies right next to them when SMOTE runs before the split.
    assert d_wrong < 0.5 * d_right, (d_wrong, d_right)
