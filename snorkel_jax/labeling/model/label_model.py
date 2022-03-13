import logging
import random
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import jax.numpy as jnp
import jax
import optax
from tqdm import trange

from snorkel_jax.labeling.analysis import LFAnalysis
#from snorkel_jax.labeling.model.base_labeler import BaseLabeler
from snorkel_jax.utils.core import probs_to_preds
from snorkel_jax.labeling.model.graph_utils import get_clique_tree
from snorkel_jax.labeling.model.loss_functions import grad_Zloss,grad_invMUloss,grad_MUloss,Zloss,invMUloss,MUloss
from snorkel_jax.analysis.scorer import Scorer
from snorkel_jax.labeling.model.logger import Logger
from snorkel_jax.types import Config
from snorkel_jax.utils.config import merge_config
from snorkel_jax.utils.lr_schedulers import LRSchedulerConfig
from snorkel_jax.utils.optimizers import OptimizerConfig
from snorkel_jax.algos import hungarian

Metrics = Dict[str, float]

class TrainConfig(Config):
    """Settings for the fit() method of LabelModel.
    Parameters
    ----------
    n_epochs
        The number of epochs to train (where each epoch is a single optimization step)
    lr
        Base learning rate (will also be affected by lr_scheduler choice and settings)
    weight_decay
        Centered L2 regularization strength
    optimizer
        Which optimizer to use (one of ["sgd", "adam", "adamax"])
    optimizer_config
        Settings for the optimizer
    lr_scheduler
        Which lr_scheduler to use (one of ["constant", "linear", "exponential", "step"])
    lr_scheduler_config
        Settings for the LRScheduler
    prec_init
        LF precision initializations / priors
    seed
        A random seed to initialize the random number generator with
    log_freq
        Report loss every this many epochs (steps)
    mu_eps
        Restrict the learned conditional probabilities to [mu_eps, 1-mu_eps]
    """

    n_epochs: int = 100
    lr: float = 0.01
    weight_decay: float = 0.0
    optimizer: str = "sgd"
    optimizer_config: OptimizerConfig = OptimizerConfig()  # type: ignore
    lr_scheduler: str = "constant"
    lr_scheduler_config: LRSchedulerConfig = LRSchedulerConfig()  # type: ignore
    prec_init: Union[float, List[float], jnp.array] = 0.7
    seed: int = 13#np.random.randint(1e6)
    log_freq: int = 10
    mu_eps: Optional[float] = None


class _CliqueData(NamedTuple):
    start_index: int
    end_index: int
    max_cliques: Set[int]

class LabelModelConfig(Config):
    """Settings for the LabelModel initialization.
    Parameters
    ----------
    verbose
        Whether to include print statements
    """

    verbose: bool = True
    seed: int = 11


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

    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None:
        self.config: LabelModelConfig = LabelModelConfig(**kwargs)
        self.cardinality = cardinality
        self.random_key = jax.random.PRNGKey(self.config.seed)

    def _set_logger(self) -> None:
        self.logger = Logger(self.train_config.log_freq)
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO)

    def _execute_logging(self, func_loss,opt_arr,additional_inputs,metrics_dict) -> Metrics:
        #only compute loss when logging
        if self.logger.check():
            # Always add average loss
            loss=func_loss(opt_arr,**additional_inputs)
            metrics_dict = {"train/loss": float(loss)}
            if self.config.verbose:
                self.logger.log(metrics_dict)

        return metrics_dict

    def _set_optimizer(self) -> None:
        optimizer_config = self.train_config.optimizer_config
        optimizer_name = self.train_config.optimizer

        if optimizer_name == "sgd":
            optimizer=optax.sgd(learning_rate=self.lr_scheduler,**optimizer_config.sgd_config._asdict())
        elif optimizer_name == "adam":
            optimizer=optax.adam(learning_rate=self.lr_scheduler,**optimizer_config.adam_config._asdict())
        elif optimizer_name == "rmsprop":
            optimizer=optax.rmsprop(learning_rate=self.lr_scheduler,**optimizer_config.rmsprop_config._asdict())
        else:
            raise ValueError(f"Unrecognized optimizer option '{optimizer_name}'")


        self.optimizer = optax.chain(
            optimizer,
            optax.additive_weight_decay(weight_decay=self.train_config.weight_decay)
        )

    def _set_lr_scheduler(self) -> None:
        # Set warmup scheduler
        self._set_warmup_scheduler()

        # Set lr scheduler
        lr_scheduler_name = self.train_config.lr_scheduler
        lr_scheduler_config = self.train_config.lr_scheduler_config
        #lr_scheduler: Optional[optim.lr_scheduler._LRScheduler]

        if lr_scheduler_name == "constant":
            lr_scheduler = self.train_config.lr
        elif lr_scheduler_name == "linear":
            lr_scheduler = optax.linear_schedule(self.train_config.lr,lr_scheduler_config.linear_config.end_value, self.n_epochs, transition_begin=0)
        elif lr_scheduler_name == "exponential":
            lr_scheduler=optax.exponential_decay(self.train_config.lr, self.n_epochs, **lr_scheduler_config.exponential_config._asdict())
        else:
            raise ValueError(f"Unrecognized lr scheduler option '{lr_scheduler_name}'")

        if self.warmup_scheduler:
            self.lr_scheduler=optax.join_schedules([self.warmup_scheduler,lr_scheduler],[self.warmup_steps])
        else:
            self.lr_scheduler = lr_scheduler

    def _set_warmup_scheduler(self) -> None:
        if self.train_config.lr_scheduler_config.warmup_steps!=0:
            warmup_steps = self.train_config.lr_scheduler_config.warmup_steps
            if warmup_steps < 0:
                raise ValueError("warmup_steps much greater or equal than 0.")
            self.warmup_steps = int(warmup_steps)
            self.n_epochs=self.train_config.n_epochs
            warmup_scheduler = optax.linear_schedule(0, self.train_config.lr, self.warmup_steps, transition_begin=0)
            if self.config.verbose:  # pragma: no cover
                logging.info(f"Warmup {self.warmup_steps} steps.")

        elif self.train_config.lr_scheduler_config.warmup_percentage:
            warmup_percentage = self.train_config.lr_scheduler_config.warmup_percentage
            self.warmup_steps = int(warmup_percentage * self.train_config.n_epochs)
            self.n_epochs=self.train_config.n_epochs-self.warmup_steps
            warmup_scheduler = optax.linear_schedule(0, self.train_config.lr, self.warmup_steps, transition_begin=0)
            if self.config.verbose:  # pragma: no cover
                logging.info(f"Warmup {self.warmup_steps} steps.")

        else:
            warmup_scheduler = None
            self.n_epochs=self.train_config.n_epochs
            self.warmup_steps = 0

        self.warmup_scheduler = warmup_scheduler
        
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
        if type(self.train_config.prec_init)==float:
            prec_init = self.train_config.prec_init * jnp.ones(self.m)
        else:
            prec_init = jnp.array(self.train_config.prec_init) * jnp.ones(self.m)

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

    def _clip_params(self) -> None:
        """Clamp the values of the learned parameter vector.
        Clamp the entries of self.mu to be in [mu_eps, 1 - mu_eps], where mu_eps is
        either set by the user, or defaults to 1 / 10 ** np.ceil(np.log10(self.n)).
        Note that if mu_eps is set too high, e.g. in sparse settings where LFs
        mostly abstain, this will result in learning conditional probabilities all
        equal to mu_eps (and/or 1 - mu_eps)!  See issue #1422.
        Note: Use user-provided value of mu_eps in train_config, else default to
            mu_eps = 1 / 10 ** np.ceil(np.log10(self.n))
        this rounding is done to make it more obvious when the parameters have been
        clamped.
        """
        if self.train_config.mu_eps is not None:
            mu_eps = self.train_config.mu_eps
        else:
            mu_eps = min(0.01, 1 / 10 ** jnp.ceil(jnp.log10(self.n)))
        self.mu = jnp.clip(self.mu,a_min=mu_eps, a_max=1 - mu_eps)  # type: ignore

    def _train_model(self,func_loss,func_grad,opt_arr,additional_inputs):
        progress_bar=True
        start_iteration=0

        # Set training components
        self._set_lr_scheduler()
        self._set_optimizer()
        
        opt_state = self.optimizer.init(opt_arr)

        n_epochs=self.warmup_steps+self.n_epochs
        if progress_bar:
            epochs = trange(start_iteration, n_epochs, unit="epoch")
        else:
            epochs = range(start_iteration, n_epochs)

        metrics_hist = {}  # The most recently seen value for all metrics
        metrics_dict={}

        for epoch in epochs:

            # compute the gradient
            grads = func_grad(opt_arr,**additional_inputs)
            #compute the updates
            updates, opt_state = self.optimizer.update(grads, opt_state, opt_arr)
            #apply updates to Z
            opt_arr = optax.apply_updates(opt_arr, updates)

            # Calculate metrics, log, and checkpoint as necessary
            if func_loss:
                metrics_dict = self._execute_logging(func_loss,opt_arr,additional_inputs,metrics_dict)
                metrics_hist.update(metrics_dict)

        # Cleanup progress bar if enabled
        if progress_bar:
            epochs.close()
        return opt_arr


    def _break_col_permutation_symmetry(self) -> None:
        r"""Heuristically choose amongst (possibly) several valid mu values.
        If there are several values of mu that equivalently satisfy the optimization
        objective, as there often are due to column permutation symmetries, then pick
        the solution that trusts the user-written LFs most.
        In more detail, suppose that mu satisfies (minimizes) the two loss objectives:
            1. O = mu @ P @ mu.T
            2. diag(O) = sum(mu @ P, axis=1)
        Then any column permutation matrix Z that commutes with P will also equivalently
        satisfy these objectives, and thus is an equally valid (symmetric) solution.
        Therefore, we select the solution that maximizes the summed probability of the
        LFs being accurate when not abstaining.
            \sum_lf \sum_{y=1}^{cardinality} P(\lf = y, Y = y)
        """
        #P represents class balance
        #mu is the learned accuracies
        d, k = self.mu.shape
        # We want to maximize the sum of diagonals of matrices for each LF. So
        # we start by computing the sum of conditional probabilities here.
        probs_sum = sum([self.mu[i : i + k] for i in range(0, self.m * k, k)]) @ self.P
        cost_mat=-probs_sum

        Z = jnp.zeros([k, k])

        # Compute groups of indicess with equal prior in P.
        P_rounded=jnp.around(self.P.diagonal(),3)
        for val_i in jnp.unique(P_rounded):
            group_bool=P_rounded==val_i
            if jnp.sum(group_bool)==1:
                Z=Z.at[group_bool,group_bool].set(1)
                continue

            # Use the Munkres algorithm to find the optimal permutation.
            # We use minus because we want to maximize diagonal sum, not minimize,
            # and transpose because we want to permute columns, not rows.
            res_mat=cost_mat[group_bool][:,group_bool]
            Z_mask=jnp.zeros(Z.shape)
            Z_mask=Z_mask.at[group_bool].set(Z_mask[group_bool]+1)
            Z_mask=Z_mask.at[:,group_bool].set(Z_mask[:,group_bool]+1)
            Z_mask=Z_mask==2
            print(Z_mask)
            print(res_mat)
            assignment_mat=hungarian.solve(res_mat)
            print(assignment_mat.astype(int))
            Z=Z.at[Z_mask].set(assignment_mat.astype(int).flatten())
            print(Z)

        # Set mu according to permutation
        print(self.mu)
        self.mu=self.mu @ Z 
        print(self.mu)

    def get_conditional_probs(self) -> jnp.array:
        r"""Return the estimated conditional probabilities table given parameters mu.
        Given a parameter vector mu, return the estimated conditional probabilites
        table cprobs, where cprobs is an (m, k+1, k)-dim np.ndarray with:
            cprobs[i, j, k] = P(\lf_i = j-1 | Y = k)
        where m is the number of LFs, k is the cardinality, and cprobs includes the
        conditional abstain probabilities P(\lf_i = -1 | Y = y).
        Parameters
        ----------
        mu
            An [m * k, k] np.ndarray with entries in [0, 1]
        Returns
        -------
        np.ndarray
            An [m, k + 1, k] np.ndarray conditional probabilities table.
        """
        cprobs = jnp.zeros((self.m, self.cardinality + 1, self.cardinality))
        for i in range(self.m):
            # si = self.c_data[(i,)]['start_index']
            # ei = self.c_data[(i,)]['end_index']
            # mu_i = mu[si:ei, :]
            mu_i = self.mu[i * self.cardinality : (i + 1) * self.cardinality, :]
            cprobs=cprobs.at[i, 1:, :].set(mu_i)

            # The 0th row (corresponding to abstains) is the difference between
            # the sums of the other rows and one, by law of total probability
            cprobs=cprobs.at[i, 0, :].set(1 - mu_i.sum(axis=0))
        return cprobs


    def get_weights(self) -> jnp.array:
        """Return the vector of learned LF weights for combining LFs.
        Returns
        -------
        np.ndarray
            [m,1] vector of learned LF weights for combining LFs.
        Example
        -------
        >>> L = np.array([[1, 1, 1], [1, 1, -1], [-1, 0, 0], [0, 0, 0]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L, seed=123)
        >>> np.around(label_model.get_weights(), 2)  # doctest: +SKIP
        array([0.99, 0.99, 0.99])
        """
        accs = jnp.zeros(self.m)
        cprobs = self.get_conditional_probs()
        for i in range(self.m):
            accs=accs.at[i].set(jnp.diag(cprobs[i, 1:, :] @ self.P).sum())
        return jnp.clip(accs / self.coverage, 1e-6, 1.0)


        
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
            weight_decay
                Weight decay parameter, default is 0.0
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
        self.train_config: TrainConfig = merge_config(  # type:ignore
            TrainConfig(), kwargs  # type:ignore
        )
        self.random_key = jax.random.PRNGKey(self.config.seed)

        # Set Logger
        self._set_logger()

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
        lf_analysis = LFAnalysis(L_train)
        self.coverage = lf_analysis.lf_coverages()

        # Compute O and initialize params
        if self.config.verbose:  # pragma: no cover
            logging.info("Computing O...")
        self._generate_O(L_shift)
        
        # Estimate \mu
        if self.config.verbose:  # pragma: no cover
            logging.info(r"Estimating \mu...")
        self._init_mu()

        if self.config.verbose:  # pragma: no cover
            logging.info("Training Model...")

        #if label functions are not independent
        if self.lf_structure:
            cond_O=jnp.linalg.cond(self.O)
            if cond_O>1000:
                print('WARNING: O may be ill-conditioned\n this may be the case if multiple labelling functions have full coverage')
            self.O_inv = jnp.linalg.inv(self.O)
            
            #estimate Z
            self.Z = jax.random.normal(self.random_key,(self.d, self.cardinality))
            self.Z = self._train_model(None,grad_Zloss,self.Z,{'O_inv':self.O_inv,'mask':self.mask})
            
            #compute Q (as estimated from Z) = \mu P \mu^T
            self._compute_Q()
    
            #estimate mu
            self.mu = self._train_model(invMUloss,grad_invMUloss,self.mu,{'Q':self.Q,'P':self.P,'O':self.O,'mask':self.mask})
            
        #label functions are independent
        else:
            self.mu = self._train_model(MUloss,grad_MUloss,self.mu,{'O':self.O,'P':self.P,'mask':self.mask})

        
        # Post-processing operations on mu
        self._clip_params()
        self._break_col_permutation_symmetry()

        # Print confusion matrix if applicable
        if self.config.verbose:  # pragma: no cover
            logging.info("Finished Training")