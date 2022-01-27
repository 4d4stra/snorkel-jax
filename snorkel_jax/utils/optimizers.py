from typing import Tuple

from snorkel_jax.types import Config


class SGDOptimizerConfig(Config):
    """Settings for SGD optimizer."""

    momentum: float = 0.9
    nesterov: bool = False


class AdamOptimizerConfig(Config):
    """Settings for Adam optimizer."""

    b1: float=0.9
    b2: float=0.999
    eps: float=1e-08
    eps_root: float=0.0


class RMSPropOptimizerConfig(Config):
    """Settings for RMSProp optimizer."""

    momentum: float = 0.9
    nesterov: bool = False
    eps: float = 1e-8
    decay: float = 0.9
    initial_scale: float = 0.0
    centered: bool=False


class OptimizerConfig(Config):
    """Settings common to all optimizers."""

    sgd_config: SGDOptimizerConfig = SGDOptimizerConfig()  # type:ignore
    adam_config: AdamOptimizerConfig = AdamOptimizerConfig()  # type:ignore
    rmsprop_config: RMSPropOptimizerConfig = RMSPropOptimizerConfig()  # type:ignore