import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy.

    Parameters:
    y_true: np.ndarray. The true label.
    y_pred: np.ndarray. The predicted label.

    Returns:
    float. The accuracy.
    """
    return np.mean(y_true == y_pred)