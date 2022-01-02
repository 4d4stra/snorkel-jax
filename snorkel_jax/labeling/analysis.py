from collections import OrderedDict
from itertools import product
from typing import List, Optional, Union

import jax.numpy as jnp
#import scipy.sparse as sparse
from pandas import DataFrame, Series
#from sklearn.metrics import confusion_matrix
from snorkel_jax.utils.core import confusion_matrix

from snorkel_jax.utils.core import to_int_label_array

#from snorkel_jax.labeling.lf.core import LabelingFunction


class LFAnalysis:
    """Run analyses on LFs using label matrix.
    Parameters
    ----------
    L
        Label matrix where L_{i,j} is the label given by the jth LF to the ith
        candidate (using -1 for abstain)
    lfs
        Labeling functions used to generate ``L``
    Raises
    ------
    ValueError
        If number of LFs and number of LF matrix columns differ
    Attributes
    ----------
    L
        See above.
    """

    def __init__(
        self, L: jnp.array, lfs: Optional[List[LabelingFunction]] = None
    ) -> None:
        self.L = L
        #self._L_sparse = sparse.csr_matrix(L + 1)
        self._lf_names = None
        if lfs is not None:
            if len(lfs) != self._L_sparse.shape[1]:
                raise ValueError(
                    f"Number of LFs ({len(lfs)}) and number of "
                    f"LF matrix columns ({self._L_sparse.shape[1]}) are different"
                )
            self._lf_names = [lf.name for lf in lfs]

    def _covered_data_points(self) -> jnp.array:
        """Get indicator vector z where z_i = 1 if x_i is labeled by at least one LF."""
        return jnp.where((self.L>=0).sum(axis=1) != 0, 1, 0)

    def _overlapped_data_points(self) -> jnp.array:
        """Get indicator vector z where z_i = 1 if x_i is labeled by more than one LF."""
        return jnp.where((self.L>=0).sum(axis=1) > 1, 1, 0)

    def _conflicted_data_points(self) -> jnp.array:
        """Get indicator vector z where z_i = 1 if x_i is labeled differently by two LFs."""
        maxes=self.L.max(axis=1)
        mins=jnp.where(self.L==-1,self.L.max(axis=1).reshape((len(self.L),1)),self.L).min(axis=1)
        return jnp.where(maxes==mins,0,1)

    def label_coverage(self) -> float:
        """Compute the fraction of data points with at least one label.
        Returns
        -------
        float
            Fraction of data points with labels
        Example
        -------
        >>> L = np.array([
        ...     [-1, 0, 0],
        ...     [-1, -1, -1],
        ...     [1, 0, -1],
        ...     [-1, 0, -1],
        ...     [0, 0, 0],
        ... ])
        >>> LFAnalysis(L).label_coverage()
        0.8
        """
        return self._covered_data_points().sum() / self.L.shape[0]

    def label_overlap(self) -> float:
        """Compute the fraction of data points with at least two (non-abstain) labels.
        Returns
        -------
        float
            Fraction of data points with overlapping labels
        Example
        -------
        >>> L = np.array([
        ...     [-1, 0, 0],
        ...     [-1, -1, -1],
        ...     [1, 0, -1],
        ...     [-1, 0, -1],
        ...     [0, 0, 0],
        ... ])
        >>> LFAnalysis(L).label_overlap()
        0.6
        """
        return self._overlapped_data_points().sum() / self.L.shape[0]

    def label_conflict(self) -> float:
        """Compute the fraction of data points with conflicting (non-abstain) labels.
        Returns
        -------
        float
            Fraction of data points with conflicting labels
        Example
        -------
        >>> L = np.array([
        ...     [-1, 0, 0],
        ...     [-1, -1, -1],
        ...     [1, 0, -1],
        ...     [-1, 0, -1],
        ...     [0, 0, 0],
        ... ])
        >>> LFAnalysis(L).label_conflict()
        0.2
        """
        return self._conflicted_data_points().sum() / self.L.shape[0]

    def lf_polarities(self) -> List[List[int]]:
        """Infer the polarities of each LF based on evidence in a label matrix.
        Returns
        -------
        List[List[int]]
            Unique output labels for each LF
        Example
        -------
        >>> L = np.array([
        ...     [-1, 0, 0],
        ...     [-1, -1, -1],
        ...     [1, 0, -1],
        ...     [-1, 0, -1],
        ...     [0, 0, 0],
        ... ])
        >>> LFAnalysis(L).lf_polarities()
        [[0, 1], [0], [0]]
        """
        return [
            sorted(jnp.unique(self.L[self.L[:,i]>=0,i]).tolist())
            for i in range(self.L.shape[1])
        ]

    def lf_coverages(self) -> jnp.array:
        """Compute frac. of examples each LF labels.  Note, -1 are abstains.
        Returns
        -------
        numpy.ndarray
            Fraction of labeled examples for each LF
        Example
        -------
        >>> L = np.array([
        ...     [-1, 0, 0],
        ...     [-1, -1, -1],
        ...     [1, 0, -1],
        ...     [-1, 0, -1],
        ...     [0, 0, 0],
        ... ])
        >>> LFAnalysis(L).lf_coverages()
        array([0.4, 0.8, 0.4])
        """
        return (self.L >= 0).sum(axis=0) / self.L.shape[0]

    def lf_overlaps(self, normalize_by_coverage: bool = False) -> jnp.array:
        """Compute frac. of examples each LF labels that are labeled by another LF.
        An overlapping example is one that at least one other LF returns a
        (non-abstain) label for.
        Note that the maximum possible overlap fraction for an LF is the LF's
        coverage, unless ``normalize_by_coverage=True``, in which case it is 1.
        Parameters
        ----------
        normalize_by_coverage
            Normalize by coverage of the LF, so that it returns the percent of LF labels
            that have overlaps.
        Returns
        -------
        numpy.ndarray
            Fraction of overlapping examples for each LF
        Example
        -------
        >>> L = np.array([
        ...     [-1, 0, 0],
        ...     [-1, -1, -1],
        ...     [1, 0, -1],
        ...     [-1, 0, -1],
        ...     [0, 0, 0],
        ... ])
        >>> LFAnalysis(L).lf_overlaps()
        array([0.4, 0.6, 0.4])
        >>> LFAnalysis(L).lf_overlaps(normalize_by_coverage=True)
        array([1.  , 0.75, 1.  ])
        """
        overlaps = (
            (self.L >= 0).T
            @ self._overlapped_data_points()
            / self.L.shape[0]
        )
        if normalize_by_coverage:
            overlaps /= self.lf_coverages()
        return jnp.nan_to_num(overlaps)

    def lf_conflicts(self, normalize_by_overlaps: bool = False) -> jnp.array:
        """Compute frac. of examples each LF labels and labeled differently by another LF.
        A conflicting example is one that at least one other LF returns a
        different (non-abstain) label for.
        Note that the maximum possible conflict fraction for an LF is the LF's
        overlaps fraction, unless ``normalize_by_overlaps=True``, in which case it is 1.
        Parameters
        ----------
        normalize_by_overlaps
            Normalize by overlaps of the LF, so that it returns the percent of LF
            overlaps that have conflicts.
        Returns
        -------
        numpy.ndarray
            Fraction of conflicting examples for each LF
        Example
        -------
        >>> L = np.array([
        ...     [-1, 0, 0],
        ...     [-1, -1, -1],
        ...     [1, 0, -1],
        ...     [-1, 0, -1],
        ...     [0, 0, 0],
        ... ])
        >>> LFAnalysis(L).lf_conflicts()
        array([0.2, 0.2, 0. ])
        >>> LFAnalysis(L).lf_conflicts(normalize_by_overlaps=True)
        array([0.5       , 0.33333333, 0.        ])
        """
        conflicts = (
            (self.L >= 0).T
            @ self._conflicted_data_points()
            / self.L.shape[0]
        )
        if normalize_by_overlaps:
            conflicts /= self.lf_overlaps()
        return jnp.nan_to_num(conflicts)

    def lf_empirical_accuracies(self, Y: jnp.array) -> jnp.array:
        """Compute empirical accuracy against a set of labels Y for each LF.
        Usually, Y represents development set labels.
        Parameters
        ----------
        Y
            [n] or [n, 1] np.ndarray of gold labels
        Returns
        -------
        numpy.ndarray
            Empirical accuracies for each LF
        """
        Y = to_int_label_array(Y)
        X = jnp.where(
            self.L == -1,
            0,
            jnp.where(self.L == Y.reshape((len(self.L),1)), 1, -1),
        )
        with jnp.errstate(divide="ignore", invalid="ignore"):
            return jnp.nan_to_num(0.5 * (X.sum(axis=0) / (self.L != -1).sum(axis=0) + 1))

    def lf_empirical_probs(self, Y: jnp.array, k: int) -> jnp.array:
        """Estimate conditional probability tables for each LF.
        Computes conditional probability tables, P(L | Y), for each LF using
        the provided true labels Y.
        Parameters
        ----------
        Y
            The n-dim array of true labels in {1,...,k}
        k
            The cardinality i.e. number of classes
        Returns
        -------
        np.ndarray
            An m x (k+1) x k np.ndarray representing the m (k+1) x k conditional probability
            tables P_i, where P_i[l,y] represents P(LF_i = l | Y = y) empirically calculated
        """
        n, m = self.L.shape

        Y = to_int_label_array(Y)

        # Compute empirical conditional probabilities
        # Note: Can do this more efficiently...
        P = jnp.zeros((m, k + 1, k))
        for y in range(k):
            is_y = jnp.where(Y == y, 1, 0)
            for j, l in product(range(m), range(-1, k)):
                P[j, l + 1, y] = jnp.where(self.L[:, j] == l, 1, 0) @ is_y / is_y.sum()
        return P

    def lf_summary(
        self, Y: Optional[jnp.array] = None, est_weights: Optional[jnp.array] = None
    ) -> DataFrame:
        """Create a pandas DataFrame with the various per-LF statistics.
        Parameters
        ----------
        Y
            [n] or [n, 1] np.ndarray of gold labels. If provided, the empirical weight
            for each LF will be calculated.
        est_weights
            Learned weights for each LF
        Returns
        -------
        pandas.DataFrame
            Summary statistics for each LF
        """
        n, m = self.L.shape
        lf_names: Union[List[str], List[int]]
        d: OrderedDict[str, Series] = OrderedDict()
        if self._lf_names is not None:
            d["j"] = list(range(m))
            lf_names = self._lf_names
        else:
            lf_names = list(range(m))

        # Default LF stats
        d["Polarity"] = Series(data=self.lf_polarities(), index=lf_names)
        d["Coverage"] = Series(data=self.lf_coverages(), index=lf_names)
        d["Overlaps"] = Series(data=self.lf_overlaps(), index=lf_names)
        d["Conflicts"] = Series(data=self.lf_conflicts(), index=lf_names)

        if Y is not None:
            #k=len(jnp.unique(Y))

            labels = jnp.unique(
                jnp.concatenate((Y.flatten(), self.L.flatten(), jnp.array([-1])))
            )
            labels=labels[1:]#remove the abstains
            confusions = [
                confusion_matrix(Y, self.L[:, i], labels) for i in range(m)
            ]
            corrects = [jnp.diagonal(conf).sum() for conf in confusions]
            incorrects = [
                conf.sum() - correct for conf, correct in zip(confusions, corrects)
            ]
            accs = self.lf_empirical_accuracies(Y)
            d["Correct"] = Series(data=corrects, index=lf_names)
            d["Incorrect"] = Series(data=incorrects, index=lf_names)
            d["Emp. Acc."] = Series(data=accs, index=lf_names)

        if est_weights is not None:
            d["Learned Weight"] = Series(est_weights, index=lf_names)

        return DataFrame(data=d, index=lf_names)