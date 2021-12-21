
from typing import Dict, List
import jax.numpy as jnp
import jax


def probs_to_preds(
    probs: jnp.array, tie_break_policy: str = "random", tol: float = 1e-5, random_seed: int = 11
) -> jnp.array:
    """Convert an array of probabilistic labels into an array of predictions.
    Policies to break ties include:
    "abstain": return an abstain vote (-1)
    "random": randomly choose among tied option using random selection
    Parameters
    ----------
    prob
        A [num_datapoints, num_classes] array of probabilistic labels such that each
        row sums to 1.
    tie_break_policy
        Policy to break ties when converting probabilistic labels to predictions
    tol
        The minimum difference among probabilities to be considered a tie
    Returns
    -------
    np.ndarray
        A [n] array of predictions (integers in [0, ..., num_classes - 1])
    Examples
    --------
    >>> probs_to_preds(np.array([[0.5, 0.5, 0.5]]), tie_break_policy="abstain")
    array([-1])
    >>> probs_to_preds(np.array([[0.8, 0.1, 0.1]]))
    array([0])
    """
    num_datapoints, num_classes = probs.shape
    if num_classes <= 1:
        raise ValueError(
            f"probs must have probabilities for at least 2 classes. "
            f"Instead, got {num_classes} classes."
        )

    Y_pred = jnp.zeros(num_datapoints)-1
    diffs = jnp.abs(probs - probs.max(axis=1).reshape(-1, 1))

    #for each row, compute how many are within tol
    count_maxIDs=jnp.sum(diffs < tol ,axis=1)

    #setting tie breakers
    if tie_break_policy == "random":
        Y_pred=Y_pred.at[count_maxIDs>1].set(jax.random.choice(jax.random.PRNGKey(random_seed),jnp.arange(num_classes),shape=(jnp.sum(count_maxIDs>1),)))
    elif tie_break_policy == "abstain":
        Y_pred=Y_pred.at[count_maxIDs>1].set(-1)
    else:
        raise ValueError(
            f"tie_break_policy={tie_break_policy} policy not recognized."
        )

    #setting classes
    for i in range(num_classes):
        Y_pred=Y_pred.at[(count_maxIDs==1) & (diffs[:,i] < tol)].set(i)

    return Y_pred.astype(int)