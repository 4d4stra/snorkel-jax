import logging
import random
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import jax.numpy as jnp
import jax
import optax
from tqdm import trange

#from snorkel.labeling.analysis import LFAnalysis
#from snorkel_jax.labeling.model.base_labeler import BaseLabeler
from snorkel_jax.utils.core import probs_to_preds
from snorkel_jax.labeling.model.graph_utils import get_clique_tree
from snorkel_jax.labeling.model.loss_functions import grad_Zloss,grad_invMUloss,grad_MUloss
from snorkel_jax.analysis.scorer import Scorer
#from snorkel.labeling.model.logger import Logger
#from snorkel.types import Config
#from snorkel.utils.config_utils import merge_config
#from snorkel.utils.lr_schedulers import LRSchedulerConfig
#from snorkel.utils.optimizers import OptimizerConfig

class _CliqueData(NamedTuple):
    start_index: int
    end_index: int
    max_cliques: Set[int]


class LabelModel:
    r"""A model for learning the LF accuracies and combining their output labels.
    This class learns a model of the labeling functions' conditional probabilities
    of outputting the true (unobserved) label `Y`, `P(\lf | Y)`, and uses this learned
    model to re-weight and combine their output labels.
    This class is based on the approach in [Training Complex Models with Multi-Task
    Weak Supervision](https://arxiv.org/abs/1810.02840), published in AAAI'19. In this
    approach, we compute the inverse generalized covariance matrix of the junction tree
    of a given LF dependency graph, and perform a matrix completion-style approach with
    respect to these empirical statistics. The result is an estimate of the conditional
    LF probabilities, `P(\lf | Y)`, which are then set as the parameters of the label
    model used to re-weight and combine the labels output by the LFs.
    Currently this class uses a conditionally independent label model, in which the LFs
    are assumed to be conditionally independent given `Y`.
    Examples
    --------
    >>> label_model = LabelModel()
    >>> label_model = LabelModel(cardinality=3)
    >>> label_model = LabelModel(cardinality=3, device='cpu')
    >>> label_model = LabelModel(cardinality=3)
    Parameters
    ----------
    cardinality
        Number of classes, by default 2
    **kwargs
        Arguments for changing config defaults
    Raises
    ------
    ValueError
        If config device set to cuda but only cpu is available
    Attributes
    ----------
    cardinality
        Number of classes, by default 2
    config
        Training configuration
    seed
        Random seed
    """

    def __init__(self, cardinality: int = 2, random_seed: int = 13, **kwargs: Any) -> None:
        #self.config: LabelModelConfig = LabelModelConfig(**kwargs)
        self.cardinality = cardinality
        self.random_key = jax.random.PRNGKey(random_seed)
        
    def _set_constants(self, L: jnp.array) -> None:
        self.n, self.m = L.shape
        if self.m < 3:
            raise ValueError("L_train should have at least 3 labeling functions")
        self.t = 1
        

    def _set_class_balance(
        self, class_balance: Optional[List[float]], Y_dev: jnp.array
    ) -> None:
        """Set a prior for the class balance.
        In order of preference:
        1) Use user-provided class_balance
        2) Estimate balance from Y_dev
        3) Assume uniform class distribution
        """
        if class_balance is not None:
            self.p = jnp.array(class_balance)
            if len(self.p) != self.cardinality:
                raise ValueError(
                    f"class_balance has {len(self.p)} entries. Does not match LabelModel cardinality {self.cardinality}."
                )
        elif Y_dev is not None:
            class_counts = Counter(Y_dev)
            sorted_counts = jnp.array([v for k, v in sorted(class_counts.items())])
            self.p = sorted_counts / sum(sorted_counts)
            if len(self.p) != self.cardinality:
                raise ValueError(
                    f"Y_dev has {len(self.p)} class(es). Does not match LabelModel cardinality {self.cardinality}."
                )
        else:
            self.p = (1 / self.cardinality) * jnp.ones(self.cardinality)

        if jnp.any(self.p == 0):
            raise ValueError(
                f"Class balance prior is 0 for class(es) {jnp.where(self.p)[0]}."
            )
        self.P = jnp.diag(self.p)
        
    def _build_mask(self):
        """Build mask applied to O^{-1}, O for the matrix approx constraint."""
        self.mask = jnp.ones((self.m * self.cardinality, self.m * self.cardinality)).astype(bool)
        for ci in self.c_data.values():
            si = ci.start_index
            ei = ci.end_index
            for cj in self.c_data.values():
                sj, ej = cj.start_index, cj.end_index

                # Check if ci and cj are part of the same maximal clique
                # If so, mask out their corresponding blocks in O^{-1}
                if len(ci.max_cliques.intersection(cj.max_cliques)) > 0:
                    self.mask=self.mask.at[si:ei, sj:ej].set(0)
                    self.mask=self.mask.at[sj:ej, si:ei].set(0)
        
    def _set_dependencies(self, deps):
        nodes = range(self.m)
        self.deps = deps
        self.c_tree = get_clique_tree(nodes, deps)
        
        # Create a helper data structure which maps cliques (as tuples of member
        # sources) --> {start_index, end_index, maximal_cliques}, where
        # the last value is a set of indices in this data structure
        self.c_data: Dict[int, _CliqueData] = {}
        for i in range(self.m):
            self.c_data[i] = _CliqueData(
                start_index=i * self.cardinality,
                end_index=(i + 1) * self.cardinality,
                max_cliques=set(
                    [
                        j
                        for j in self.c_tree.nodes()
                        if i in self.c_tree.nodes[j]["members"]
                    ]
                ),
            )
        
        self._build_mask()
        
        # if true, use the inverse form
        # if false, assume conditionally independent labelling functions
        self.lf_structure = len(self.deps) > 0
        
    # TODO: higher order data was never fully implemented into this
    def _get_augmented_label_matrix(self,L):
        L_aug= jnp.zeros((self.n, self.m * self.cardinality))
        for y in range(1, self.cardinality + 1):
            # A[x::y] slices A starting at x at intervals of y
            # e.g., np.arange(9)[0::3] == np.array([0,3,6])
            L_aug=L_aug.at[:, (y - 1) :: self.cardinality].set(jnp.where(L == y, 1, 0))  
        return L_aug
        
    def _generate_O(self,L):

        L_aug=self._get_augmented_label_matrix(L)

        self.d = L_aug.shape[1]

        self.O = (L_aug.T @ L_aug) / self.n
        
        
    def _compute_Q(self):
        # TODO: where does this equation come from?
        I_k = jnp.eye(self.cardinality)
        self.Q = self.O @ self.Z @ jnp.linalg.inv(I_k + self.Z.T @ self.O @ self.Z) @ self.Z.T @ self.O
        
        
    def _init_mu(self):
        r"""Initialize the learned params.
        - \mu is the primary learned parameter, where each row corresponds to
        the probability of a clique C emitting a specific combination of labels,
        conditioned on different values of Y (for each column); that is:
            self.mu[i*self.cardinality + j, y] = P(\lambda_i = j | Y = y)
        and similarly for higher-order cliques.
        Raises
        ------
        ValueError
            If prec_init shape does not match number of LFs
        """
        # Initialize mu so as to break basic reflective symmetry
        # Note that we are given either a single or per-LF initial precision
        # value, prec_i = P(Y=y|\lf=y), and use:
        #   mu_init = P(\lf=y|Y=y) = P(\lf=y) * prec_i / P(Y=y)

        # Handle single values
        prec_init = 0.7 * jnp.ones(self.m)

        # Get the per-value labeling propensities
        # Note that self.O must have been computed already!
        lps = jnp.diag(self.O)

        # TODO: Update for higher-order cliques!
        mu_init = jnp.zeros((self.d, self.cardinality))
        for i in range(self.m):
            for y in range(self.cardinality):
                idx = i * self.cardinality + y
                mu_init=mu_init.at[idx,y].set(jax.lax.clamp(0.,lps[idx] * prec_init[i] / self.p[y], 1.))

        # Initialize randomly based on self.mu_init
        self.mu = mu_init.clone() * jax.random.uniform(self.random_key)

    def _train_model(self,func_loss,opt_arr,additional_inputs):
        n_epochs=500
        progress_bar=True
        start_iteration=0
        
        optimizer=optax.sgd(learning_rate=0.01)
        opt_state = optimizer.init(opt_arr)

        if progress_bar:
            epochs = trange(start_iteration, n_epochs, unit="epoch")
        else:
            epochs = range(start_iteration, n_epochs)


        for epoch in epochs:

            # compute the gradient
            grads = func_loss(opt_arr,**additional_inputs)
            #compute the updates
            updates, opt_state = optimizer.update(grads, opt_state)
            #apply updates to Z
            opt_arr = optax.apply_updates(opt_arr, updates)

        # Cleanup progress bar if enabled
        if progress_bar:
            epochs.close()
        return opt_arr
        
    def predict_proba(self, L: jnp.array) -> jnp.array:
        r"""Return label probabilities P(Y | \lambda).
        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}f
        Returns
        -------
        np.ndarray
            An [n,k] array of probabilistic labels
        Example
        -------
        >>> L = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L, seed=123)
        >>> np.around(label_model.predict_proba(L), 1)  # doctest: +SKIP
        array([[1., 0.],
               [0., 1.],
               [0., 1.]])
        """
        L_shift = L + 1  # convert to {0, 1, ..., k}
        self._set_constants(L_shift)
        L_aug = self._get_augmented_label_matrix(L_shift)
        jtm = jnp.ones(L_aug.shape[1])

        # Note: We omit abstains, effectively assuming uniform distribution here
        X = jnp.exp(L_aug @ jnp.diag(jtm) @ jnp.log(self.mu) + jnp.log(self.p))
        Z = jnp.tile(X.sum(axis=1).reshape(-1, 1), self.cardinality)
        return X / Z
    

    def predict(
        self,
        L: jnp.array,
        return_probs: Optional[bool] = False,
        tie_break_policy: str = "abstain",
    ) -> Union[jnp.array, Tuple[jnp.array, jnp.array]]:
        """Return predicted labels, with ties broken according to policy.
        Policies to break ties include:
        "abstain": return an abstain vote (-1)
        "random": randomly choose among tied option using deterministic hash
        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        return_probs
            Whether to return probs along with preds
        tie_break_policy
            Policy to break ties when converting probabilistic labels to predictions
        Returns
        -------
        np.ndarray
            An [n,1] array of integer labels
        (np.ndarray, np.ndarray)
            An [n,1] array of integer labels and an [n,k] array of probabilistic labels
        """
        Y_probs = self.predict_proba(L)
        Y_p = probs_to_preds(Y_probs, tie_break_policy)
        if return_probs:
            return Y_p, Y_probs
        return Y_p


    def score(
        self,
        L: jnp.array,
        Y: jnp.array,
        metrics: Optional[List[str]] = ["accuracy"],
        tie_break_policy: str = "abstain",
    ) -> Dict[str, float]:
        """Calculate one or more scores from user-specified and/or user-defined metrics.
        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y
            Gold labels associated with data points in L
        metrics
            A list of metric names
        tie_break_policy
            Policy to break ties when converting probabilistic labels to predictions
        Returns
        -------
        Dict[str, float]
            A dictionary mapping metric names to metric scores
        """
        if tie_break_policy == "abstain":  # pragma: no cover
            logging.warning(
                "Metrics calculated over data points with non-abstain labels only"
            )

        Y_pred, Y_prob = self.predict(
            L, return_probs=True, tie_break_policy=tie_break_policy
        )

        scorer = Scorer(metrics=metrics)
        results = scorer.score(Y, Y_pred, Y_prob)
        return results


    def fit(
        self,
        L_train: jnp.array,
        Y_dev: Optional[jnp.array] = None,
        class_balance: Optional[List[float]] = None,
        deps: Optional[List[Tuple[int]]]=[],
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> None:
        """Train label model.
        Train label model to estimate mu, the parameters used to combine LFs.
        Parameters
        ----------
        L_train
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y_dev
            Gold labels for dev set for estimating class_balance, by default None
        class_balance
            Each class's percentage of the population, by default None
        progress_bar
            To display a progress bar, by default True
        **kwargs
            Arguments for changing train config defaults.
            n_epochs
                The number of epochs to train (where each epoch is a single
                optimization step), default is 100
            lr
                Base learning rate (will also be affected by lr_scheduler choice
                and settings), default is 0.01
            l2
                Centered L2 regularization strength, default is 0.0
            optimizer
                Which optimizer to use (one of ["sgd", "adam", "adamax"]),
                default is "sgd"
            optimizer_config
                Settings for the optimizer
            lr_scheduler
                Which lr_scheduler to use (one of ["constant", "linear",
                "exponential", "step"]), default is "constant"
            lr_scheduler_config
                Settings for the LRScheduler
            prec_init
                LF precision initializations / priors, default is 0.7
            seed
                A random seed to initialize the random number generator with
            log_freq
                Report loss every this many epochs (steps), default is 10
            mu_eps
                Restrict the learned conditional probabilities to
                [mu_eps, 1-mu_eps], default is None
        Raises
        ------
        Exception
            If loss in NaN
        Examples
        --------
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> Y_dev = [0, 1, 0]
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> label_model.fit(L, Y_dev=Y_dev, seed=2020, lr=0.05)
        >>> label_model.fit(L, class_balance=[0.7, 0.3], n_epochs=200, l2=0.4)
        """
        # Set Logger
        ##self._set_logger()

        # shifted so that abstentions (-1) can very simply be excluded from the overlap matrix
        L_shift = L_train + 1  # convert to {0, 1, ..., k}
        if L_shift.max() > self.cardinality:
            raise ValueError(
                f"L_train has cardinality {L_shift.max()}, cardinality={self.cardinality} passed in."
            )

        # initializing
        self._set_constants(L_shift)
        self._set_class_balance(class_balance, Y_dev)
        self._set_dependencies(deps)
        ##lf_analysis = LFAnalysis(L_train)
        ##self.coverage = lf_analysis.lf_coverages()

        # Compute O and initialize params
        ##if self.config.verbose:  # pragma: no cover
        ##    logging.info("Computing O...")
        self._generate_O(L_shift)
        self._init_mu()
        
        # Estimate \mu
        ##if self.config.verbose:  # pragma: no cover
        ##    logging.info(r"Estimating \mu...")

        # Set training components
        ##self._set_optimizer()
        ##self._set_lr_scheduler()
        
        #if label functions are not independent
        if self.lf_structure:
            cond_O=jnp.linalg.cond(self.O)
            if cond_O>1000:
                print('WARNING: O may be ill-conditioned\n this may be the case if multiple labelling functions have full coverage')
            self.O_inv = jnp.linalg.inv(self.O)
            
            #estimate Z
            self.Z = jax.random.normal(self.random_key,(self.d, self.cardinality))
            self.Z = self._train_model(grad_Zloss,self.Z,{'O_inv':self.O_inv,'mask':self.mask})
            
            #compute Q (as estimated from Z) = \mu P \mu^T
            self._compute_Q()
    
            #estimate mu
            self.mu = self._train_model(grad_invMUloss,self.mu,{'Q':self.Q,'P':self.P,'O':self.O,'mask':self.mask})
            
        #label functions are independent
        else:
            self.mu = self._train_model(grad_MUloss,self.mu,{'O':self.O,'P':self.P,'mask':self.mask})

        
        # Post-processing operations on mu
        ##self._clamp_params()
        ##self._break_col_permutation_symmetry()

        # Print confusion matrix if applicable
        ##if self.config.verbose:  # pragma: no cover
        ##    logging.info("Finished Training")