import os
import numpy as np
from typing import Tuple

def datasets(file_path: str='./data/', train: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the data and label.

    Parameters:
    file_path: str. The path of the data.
    train: bool. Whether to load the train data or test data.

    Returns:
    X: np.ndarray. The data.
    y: np.ndarray. The label.

    Raises:
    FileNotFoundError: If the file is not found.
    """

    mode = 'train' if train else 'test'
    
    try:
        # Load the data and label using os.path.join for better path handling
        X = np.load(os.path.join(file_path, f"{mode}_data.npy"))
        y = np.load(os.path.join(file_path, f"{mode}_label.npy")).reshape(-1)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Expected file not found: {e.filename}") from e

    return X, y