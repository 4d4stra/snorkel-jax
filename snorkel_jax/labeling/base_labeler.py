import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp

#from snorkel.analysis import Scorer
#from snorkel.utils import probs_to_preds


class BaseLabeler(ABC):
    """Abstract baseline label voter class."""

    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None:
        self.cardinality = cardinality

    @abstractmethod
    def predict_proba(self, L: jnp.array) -> jnp.array:
        """Abstract method for predicting probabilistic labels given a label matrix.
        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}f
        Returns
        -------
        np.ndarray
            An [n,k] array of probabilistic labels
        """
        pass