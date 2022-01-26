# snorkel-jax
Additional updates from snorkel package:
Merge in handling of dependencies from MeTaL
Migrate parameter l2 => weight_decay
vectorized label model predict method
removed "true random" from tie break rules
some optimizations in the way things are computed in LFAnalysis

| Parameter | Description | Default |
| --- | --- | --- |
| n_epochs | The number of optimization steps | 100 |
| lr | The peak learning rate | 0.01 |
| weight_decay | Parameter weight decay rate | 0.0 |