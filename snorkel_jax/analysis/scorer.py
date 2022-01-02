from functools import partial
from typing import Callable, Dict, List, Mapping, Optional, Union

import jax.numpy as jnp
import pandas as pd

from snorkel_jax.analysis.metrics import METRICS, metric_score


class Scorer:
    """Calculate one or more scores from user-specified and/or user-defined metrics.
    Parameters
    ----------
    metrics
        A list of metric names, all of which are defined in METRICS
    custom_metric_funcs
        An optional dictionary mapping the names of custom metrics to the functions
        that produce them. Each custom metric function should accept golds, preds, and
        probs as input (just like the standard metrics in METRICS) and return either a
        single score (float) or a dictionary of metric names to scores (if the function
        calculates multiple values, for example). See the unit tests for an example.
    abstain_label
        The gold label for which examples will be ignored. By default, follow convention
        that abstains are -1.
    Raises
    ------
    ValueError
        If a specified standard metric is not found in the METRICS dictionary
    Attributes
    ----------
    metrics
        A dictionary mapping metric names to the corresponding functions for calculating
        that metric
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        custom_metric_funcs: Optional[Mapping[str, Callable[..., float]]] = None,
        abstain_label: Optional[int] = -1,
    ) -> None:

        self.metrics: Dict[str, Callable[..., float]]
        self.metrics = {}
        if metrics:
            for metric in metrics:
                if metric not in METRICS:
                    raise ValueError(f"Unrecognized metric: {metric}")

                filter_dict = (
                    {}
                    if abstain_label is None or metric == "coverage"
                    else {"golds": [abstain_label], "preds": [abstain_label]}
                )
                self.metrics.update(
                    {
                        metric: partial(
                            metric_score, metric=metric, filter_dict=filter_dict
                        )
                    }
                )

        if custom_metric_funcs is not None:
            self.metrics.update(custom_metric_funcs)

        self.abstain_label = abstain_label

    def score(
        self,
        golds: jnp.array,
        preds: Optional[jnp.array] = None,
        probs: Optional[jnp.array] = None,
    ) -> Dict[str, float]:
        """Calculate scores for one or more user-specified metrics.
        Parameters
        ----------
        golds
            An array of gold (int) labels to base scores on
        preds
            An [n_datapoints,] or [n_datapoints, 1] array of (int) predictions to score
        probs
            An [n_datapoints, n_classes] array of probabilistic (float) predictions
        Because most metrics require either `preds` or `probs`, but not both, these
        values are optional; it is up to the metric function that will be called  to
        raise an exception if a field it requires is not passed to the `score()` method.
        Returns
        -------
        Dict[str, float]
            A dictionary mapping metric names to metric scores
        Raises
        ------
        ValueError
            If no gold labels were provided
        """
        if len(golds) == 0:  # type: ignore
            raise ValueError("Cannot score empty labels")

        metric_dict = dict()

        for metric_name, metric in self.metrics.items():
            score = metric(golds, preds, probs)
            metric_dict[metric_name] = score

        return metric_dict
