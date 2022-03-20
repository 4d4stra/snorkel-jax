# snorkel-jax
This package takes the core ideas implemented in *[Snorkel](https://github.com/snorkel-team/snorkel)* and *[Snorkel MeTaL](https://github.com/HazyResearch/metal)* and reimplements them fully in *[JAX](https://github.com/google/jax)*.

This package follows *[Snorkel](https://github.com/snorkel-team/snorkel)* most closely, however, there are a number of changes.
1. Methodological Updates:
    1. Merge in handling of dependencies from MeTaL
2. Optimization Models:
    1. Optimization model "Adamax" was removed
    2. "RMSProp" was added
    3. Migrate parameter l2 => weight_decay
3. Implementation Optimization
    1. There is no base numpy in this implementation; everything is fully implemented with *[JAX](https://github.com/google/jax)*
    2. vectorized label model predict method
    3. some optimizations in the way things are computed in LFAnalysis
    4. Native implementation of the Hungarian Algorithm for handling column permutation symmetry
4. Miscellaneous
    1. removed "true random" from tie break rules



| Parameter | Description | Default | Valid Types |
| --- | --- | --- | --- |
| L_train | An [n,m] matrix with values in {-1,0,1,...,k-1}, where -1 means abstain | REQUIRED | jax.numpy.array |
| Y_dev | Gold labels for dev set for estimating class_balance | None | jax.numpy.array |
| class_balance | Each class's percentage of the population | None | jax.numpy.array |
| progress_bar | To display a progress bar | True | bool |
| n_epochs | The number of optimization steps | 100 | int |
| lr | The peak learning rate | 0.01 | float |
| weight_decay | Parameter weight decay rate | 0.0 | float |
| optimizer | Which optimizer to use (one of ["sgd", "adam", "rmsprop"]) | "sgd" | str |
| optimizer_config | Settings for the optimizer (see below) | | dict |
| lr_scheduler | Scheduler for adjusting the learning rate | "constant" | str |
| lr_scheduler_config | Settings for the LRScheduler (see below) | | dict |
| prec_init | LF precision initializations / priors | 0.7 | float, list |
| seed | A random seed to initialize the random number generator with; due to the way JAX generates random numbers, this will be the seed for each operation that requires random number generation | 11 | int |
| log_freq | Report loss every log_freq steps | 10 | int |
| mu_eps | Restrict the learned conditional probabilities to [mu_eps, 1-mu_eps] | min(0.01, 1 / 10 ** jnp.ceil(jnp.log10(self.n))) | float |

Optimizer Config

| Config Key | Parameter | Description | Default | Valid Types |
| --- | --- | --- | --- | --- |
| sgd_config | momentum | the decay rate used by the momentum term, when it is set to None, then momentum is not used at all | 0.9 | float, None |
| sgd_config | nesterov | whether nesterov momentum is used | False | bool |
| adam_config | b1 | the exponential decay rate to track the first moment of past gradients | 0.9 | float |
| adam_config | b2 | the exponential decay rate to track the second moment of past gradients | 0.999 | float |
| adam_config | eps | a small constant applied to denominator outside of the square root (as in the Adam paper) to avoid dividing by zero when rescaling. | 1e-8 | float |
| adam_config | eps_root | a small constant applied to denominator inside the square root (as in RMSProp), to avoid dividing by zero when rescaling. This is needed for example when computing (meta-)gradients through Adam. | 0.0 | float |
| rmsprop_config | decay | the decay used to track the magnitude of previous gradients | 0.9 | float |
| rmsprop_config | eps | a small numerical constant to avoid dividing by zero when rescaling | 1e-8 | float |
| rmsprop_config | initial_scale | initialisation of accumulators tracking the magnitude of previous updates. PyTorch uses 0, TF1 uses 1. When reproducing results from a paper, verify the value used by the authors. | 0 | float |
| rmsprop_config | centered whether the second moment or the variance of the past gradients is used to rescale the latest gradients | False | bool |
| rmsprop_config | momentum | the decay rate used by the momentum term, when it is set to None, then momentum is not used at all | 0.9 | float, None |
| rmsprop_config | nesterov | whether nesterov momentum is used | False | bool |

LR Scheduler Config

| Config Key | Parameter | Description | Default | Valid Types |
| --- | --- | --- | --- | --- |
| --- | warmup_steps | The number of warmup_units over which to perform learning rate warmup (a linear increase from 0 to the specified lr) | 0 | float |
| --- | warmup_percentage | The percentage of the training procedure to warm up over (ignored if warmup_steps is non-zero) | 0.0 | float |
| exponential_config | decay_rate | must not be zero. The decay rate. | 0.1 | float |
| exponential_config | end_value | a floor for the learning rate | None | float, None |
| linear_config | end_value | the final value of the learning rate | 0. | float |