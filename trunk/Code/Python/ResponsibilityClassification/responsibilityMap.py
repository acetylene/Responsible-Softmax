import numpy as np


def rmap(modelcoeffs: np.array, p: np.array) -> np.array:
    assert p.shape[1] == 1, f"p should be a {modelcoeffs.shape[0]} by 1 column vector"
    assert modelcoeffs.shape[0] == p.shape[
        0], f"The matrix F and vector p have incompatible dimensions. F has shape {modelcoeffs.shape} and p has shape {p.shape}"

    n = modelcoeffs.shape[1]
    denoms = 1 / (np.transpose(p) @ modelcoeffs)
    return np.transpose(np.sum((1 / n) * p * modelcoeffs * denoms, axis=1))
