import numpy as np
from typing import Callable, Literal, Tuple

class EarlyStopping(object):
    """
    EarlyStopping class is used to stop training when a monitored metric has stopped improving.
    """
    def __init__(self, criterion: Callable[[np.ndarray, np.ndarray], float], min_delta: float=0.0, patience: int=5, 
                 mode: Literal['min', 'max']='min', restore_best_weights: bool=False, start_from_epoch: int=0) -> None:
        """
        Initialize EarlyStopping object.

        Parameters:
        - criterion (Callable[[np.ndarray, np.ndarray], float]): callable function that takes two numpy arrays (y_val, y_pred) and returns a float value.
        - min_delta (float): minimum change in the monitored metric to qualify as an improvement. Default: 0.0.
        - patience (int): number of epochs with no improvement after which training will be stopped. Default: 5.
        - mode (Literal['min', 'max']): one of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; 
                in 'max' mode it will stop when the quantity monitored has stopped increasing. Default: 'min'.
        - restore_best_weights (bool): whether to restore model weights from the epoch with the best value of the monitored quantity. Default: False.
        - start_from_epoch (int): epoch number from which to start monitoring. Default: 0.

        Returns:
        - None
        """
        self.criterion = criterion
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        # Initialize pre_value according to mode
        if self.mode == 'min':
            self.prev_value = float('inf')
        elif self.mode == 'max':
            self.prev_value = float('-inf')
        # Initialize counter and step
        self.counter = 0
        self.step = 0
        self.save_best = False

    def check_improvement(self, curr_value: float) -> bool:
        """
        Check if the monitored metric has improved.

        Parameters:
        - curr_value (float): current value of the monitored metric.

        Returns:
        - bool: True if the monitored metric has improved, False otherwise.
        """
        if self.mode == 'min':
            return curr_value < self.prev_value - self.min_delta
        elif self.mode == 'max':
            return curr_value > self.prev_value + self.min_delta
    
    def __call__(self, y_val: np.ndarray, y_pred: np.ndarray) -> Tuple[bool, bool]:
        """
        Update EarlyStopping object and check if training should be stopped.

        Parameters:
        - y_val (np.ndarray): validation labels.
        - y_pred (np.ndarray): predicted labels.

        Returns:
        - Tuple[bool, bool]: (True, True) if training should be stopped and best weights should be saved, 
                             (True, False) if training should be stopped but best weights should not be saved, 
                             (False, False) otherwise.
        """
        # Update step
        curr_epoch = self.step
        self.step += 1
        # If curr_epoch is less than start_from_epoch, return False, False
        if curr_epoch <= self.start_from_epoch:
            return False, False
        # Compute monitored metric
        curr_value = self.criterion(y_val, y_pred)
        # Check if curr_value has improved
        if self.check_improvement(curr_value):
            self.counter = 0
            self.prev_value = curr_value
            if self.restore_best_weights:
                self.save_best = True
        else:
            self.counter += 1
        # Check if training should be stopped and if best weights should be saved
        return self.counter >= self.patience, self.save_best