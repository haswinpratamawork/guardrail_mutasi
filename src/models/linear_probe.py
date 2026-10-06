"""Linear-probe classifier head: frozen backbone embeddings -> logreg/XGBoost.

Fast, CPU-friendly, low overfitting risk on ~1,000 labeled pages — the
baseline every fine-tuned model must beat to justify its extra cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)

# Bumped whenever the on-disk bundle layout changes incompatibly. load_probe
# refuses anything it doesn't recognise rather than silently mis-reading a
# bundle written by a different version of this file.
ARTIFACT_FORMAT_VERSION = 1


@dataclass
class ProbeResult:
    classifier_name: str
    model: object

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.model.predict(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Binary: positive-class probability, shape (n,). Multi-class: full
        per-class probability matrix, shape (n, n_classes)."""
        proba = self.model.predict_proba(embeddings)
        return proba[:, 1] if proba.shape[1] == 2 else proba


def train_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifier: Literal["logreg", "xgboost"],
    class_weight: dict | None = None,
) -> ProbeResult:
    num_classes = len(np.unique(labels))

    if classifier == "logreg":
        model = LogisticRegression(max_iter=2000, class_weight=class_weight or "balanced")
        model.fit(embeddings, labels)
    elif classifier == "xgboost":
        # Imported here rather than at module scope: the serving container loads
        # this module for `load_probe` alone and ships only the logreg head, so
        # it shouldn't have to carry xgboost's ~150MB of native libraries.
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )
        if num_classes == 2:
            # xgboost wants scale_pos_weight, not sklearn-style class_weight dict.
            pos = int(labels.sum())
            neg = int(len(labels) - pos)
            model.set_params(scale_pos_weight=(neg / pos) if pos else 1.0)
            model.fit(embeddings, labels)
        else:
            # scale_pos_weight only applies to binary; balance multi-class via sample_weight.
            sample_weight = compute_sample_weight("balanced", labels)
            model.fit(embeddings, labels, sample_weight=sample_weight)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    return ProbeResult(classifier_name=classifier, model=model)


def save_probe(
    result: ProbeResult,
    path: str | Path,
    meta: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Serialize a fitted probe to `path` as a single self-contained bundle.

    `extra` carries anything the serving side needs alongside the head — in
    practice the frozen backbone's state_dict, so a deployed container
    reproduces the exact weights the probe was fitted on without reaching out
    to the HuggingFace hub at runtime.

    Written via a temp name and renamed, so an interrupted write can't leave a
    truncated bundle that a later load would happily trust (same reasoning as
    `_save_atomic` in export_predictions_test467.py).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "classifier_name": result.classifier_name,
        "model": result.model,
        "meta": meta,
        **(extra or {}),
    }
    tmp = path.parent / (path.name + ".partial")
    joblib.dump(payload, tmp)
    tmp.replace(path)


def load_probe(path: str | Path) -> tuple[ProbeResult, dict[str, Any]]:
    """Inverse of `save_probe`. Returns (probe, payload).

    The full payload comes back rather than just `meta` so callers can reach
    the backbone weights stored under `extra`.
    """
    payload = joblib.load(Path(path))
    got = payload.get("format_version")
    if got != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"{path} has artifact format_version {got!r}, this code expects "
            f"{ARTIFACT_FORMAT_VERSION}. Re-run persist_probe.py to regenerate it."
        )
    trained_with = payload.get("meta", {}).get("versions", {}).get("sklearn")
    if trained_with and trained_with != sklearn.__version__:
        # Unpickling an estimator across sklearn versions is not guaranteed —
        # warn loudly rather than serve predictions from a silently-degraded model.
        logger.warning(
            "Probe bundle %s was fitted with scikit-learn %s but this process has %s. "
            "Predictions may not reproduce; regenerate the bundle to be safe.",
            path, trained_with, sklearn.__version__,
        )
    return ProbeResult(classifier_name=payload["classifier_name"], model=payload["model"]), payload
