from snorkel_jax.types import Config


class ExponentialLRSchedulerConfig(Config):
    """Settings for Exponential decay learning rate scheduler."""

    decay_rate: float = 0.1
    end_value: float = None

class LinearLRSchedulerConfig(Config):
    """Settings for Linear decay learning rate scheduler."""

    end_value: float = 0.


class LRSchedulerConfig(Config):
    """Settings common to all LRSchedulers.
    Parameters
    ----------
    warmup_steps
        The number of warmup_units over which to perform learning rate warmup (a linear
        increase from 0 to the specified lr)
    warmup_percentage
        The percentage of the training procedure to warm up over (ignored if
        warmup_steps is non-zero)
    exponential_config
        Extra settings for the ExponentialLRScheduler
    linear_config
        Extra settings for the LinearLRScheduler
    """

    warmup_steps: float = 0  # warm up steps
    warmup_percentage: float = 0.0  # warm up percentage
    linear_config: LinearLRSchedulerConfig = LinearLRSchedulerConfig()  # type:ignore
    exponential_config: ExponentialLRSchedulerConfig = ExponentialLRSchedulerConfig()  # type:ignore