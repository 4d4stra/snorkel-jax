# snorkel-jax
Additional updates from snorkel package:
Merge in handling of dependencies from MeTaL
Optimization model "Adamax" was replace by "RMSProp"
Migrate parameter l2 => weight_decay
vectorized label model predict method
removed "true random" from tie break rules
some optimizations in the way things are computed in LFAnalysis

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
|<td rowspan=4>L1 Name</td>|
| Parameter | Description | Default | Valid Types |
| --- | --- | --- | --- |