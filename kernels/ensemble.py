"""Class alignment and probability pooling used by the E17 submission."""

import numpy as np
from scipy.special import log_softmax, logsumexp


def native_log_probabilities(plain, manifest_classes, classes, scales):
    """Calibrate each fold in its own label space, align columns and average probabilities."""
    labels = list(manifest_classes)
    target = list(classes)
    if len(set(labels)) != len(labels) or len(labels) != len(target) or set(labels) != set(target):
        raise ValueError("Both class orders must contain the same unique labels")
    logits = np.asarray(plain)
    scales = np.asarray(scales, dtype=float)
    if logits.ndim != 3 or logits.shape[0] == 0 or logits.shape[2] != len(target):
        raise ValueError("Expected logits of shape (folds, passages, classes)")
    if scales.shape != (logits.shape[0],) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Expected one finite, positive calibration scale per fold")
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    order = [labels.index(label) for label in target]
    aligned = logits[:, :, order]
    log_probs = log_softmax(aligned * scales[:, None, None], axis=2)
    return logsumexp(log_probs, axis=0) - np.log(len(scales)), aligned, order


def blend_e17(logP6, logP14, logPq25, logPdeb, logPnat):
    """Return E16 weighted scores and E17 scores in the same passage and label order."""
    members = [np.asarray(x) for x in (logP6, logP14, logPq25, logPdeb, logPnat)]
    if members[0].ndim != 2 or any(x.shape != members[0].shape for x in members):
        raise ValueError("All members must have the same (passages, classes) shape")
    if any(not np.isfinite(x).all() for x in members):
        raise ValueError("Non-finite member scores")
    e6, q14, q25, deb, native = members
    e16 = (2 * e6 + 3 * q14 + q25 + deb) / 7
    return e16, 0.6 * log_softmax(e16, axis=1) + 0.4 * native
