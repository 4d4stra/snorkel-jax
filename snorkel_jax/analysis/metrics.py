from typing import Any, Callable, Dict, List, NamedTuple, Optional

import jax.numpy as jnp

from snorkel_jax.utils.core import filter_labels, to_int_label_array


class Metric(NamedTuple):
    """Specification for a metric and the subset of [golds, preds, probs] it expects."""

    func: Callable[..., float]
    inputs: List[str] = ["golds", "preds"]


def metric_score(
    golds: Optional[jnp.array] = None,
    preds: Optional[jnp.array] = None,
    probs: Optional[jnp.array] = None,
    metric: str = "accuracy",
    filter_dict: Optional[Dict[str, List[int]]] = None,
    **kwargs: Any,
) -> float:
    """Evaluate a standard metric on a set of predictions/probabilities.
    Parameters
    ----------
    golds
        An array of gold (int) labels
    preds
        An array of (int) predictions
    probs
        An [n_datapoints, n_classes] array of probabilistic (float) predictions
    metric
        The name of the metric to calculate
    filter_dict
        A mapping from label set name to the labels that should be filtered out for
        that label set
    Returns
    -------
    float
        The value of the requested metric
    Raises
    ------
    ValueError
        The requested metric is not currently supported
    ValueError
        The user attempted to calculate roc_auc score for a non-binary problem
    """
    if metric not in METRICS:
        msg = f"The metric you provided ({metric}) is not currently implemented."
        raise ValueError(msg)

    # Print helpful error messages if golds or preds has invalid shape or type
    golds = to_int_label_array(golds) if golds is not None else None
    preds = to_int_label_array(preds) if preds is not None else None

    # Optionally filter out examples (e.g., abstain predictions or unknown labels)
    label_dict = {"golds": golds, "preds": preds, "probs": probs}
    if filter_dict:
        if set(filter_dict.keys()).difference(set(label_dict.keys())):
            raise ValueError(
                "filter_dict must only include keys in ['golds', 'preds', 'probs']"
            )
        label_dict = filter_labels(label_dict, filter_dict)

    # Confirm that required label sets are available
    func, label_names = METRICS[metric]
    for label_name in label_names:
        if label_dict[label_name] is None:
            raise ValueError(f"Metric {metric} requires access to {label_name}.")

    label_sets = [label_dict[label_name] for label_name in label_names]
    return func(*label_sets, **kwargs)


#def _coverage_score(preds: jnp.array) -> float:
#    return np.sum(preds != -1) / len(preds)


#def _roc_auc_score(golds: np.ndarray, probs: np.ndarray) -> float:
#    if not probs.shape[1] == 2:
#        raise ValueError(
#            "Metric roc_auc is currently only defined for binary problems."
#        )
#    return skmetrics.roc_auc_score(golds, probs[:, 1])



#def _f1_micro_score(golds: jnp.array, preds: jnp.array) -> float:
#    return skmetrics.f1_score(golds, preds, average="micro")


#def _f1_macro_score(golds: jnp.array, preds: jnp.array) -> float:
 #   return skmetrics.f1_score(golds, preds, average="macro")

def _accuracy(golds: jnp.array, preds:jnp.array) -> float:
    return jnp.sum(golds==preds)/len(golds)

def _precision(golds: jnp.array, preds:jnp.array) -> dict:
    classes=jnp.unique(golds)
    dict_out={}
    for class_i in classes:
        class_i=int(class_i)
        tp_i=jnp.sum((preds==class_i) & (preds==golds))
        fp_i=jnp.sum((preds==class_i) & (preds!=golds))
        dict_out[class_i]=tp_i/(tp_i+fp_i)
    return dict_out

def _recall(golds: jnp.array, preds:jnp.array) -> dict:
    classes=jnp.unique(golds)
    dict_out={}
    for class_i in classes:
        class_i=int(class_i)
        tp_i=jnp.sum((preds==class_i) & (preds==golds))
        fn_i=jnp.sum((golds==class_i) & (preds!=golds))
        dict_out[class_i]=tp_i/(tp_i+fn_i)
    return dict_out


def _f1_score(golds: jnp.array, preds:jnp.array) -> dict:
    classes=jnp.unique(golds)
    dict_out={}
    for class_i in classes:
        class_i=int(class_i)
        tp_i=jnp.sum((preds==class_i) & (preds==golds))
        fp_i=jnp.sum((preds==class_i) & (preds!=golds))
        fn_i=jnp.sum((golds==class_i) & (preds!=golds))
        dict_out[class_i]=2*tp_i/(2*tp_i+fp_i+fn_i)
    return dict_out

# See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# for details on the definitions and available kwargs for all metrics from scikit-learn
METRICS = {
    "accuracy": Metric(_accuracy,['golds','preds']),
    #"coverage": Metric(_coverage_score, ["preds"]),
    "precision": Metric(_precision,['golds','preds']),
    "recall": Metric(_recall,['golds','preds']),
    "f1": Metric(_f1_score, ["golds", "preds"]),
    #"matthews_corrcoef": Metric(skmetrics.matthews_corrcoef),
    #"roc_auc": Metric(_roc_auc_score, ["golds", "probs"]),
}