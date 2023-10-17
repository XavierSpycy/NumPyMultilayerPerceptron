class StepLR(object):
    def __init__(self, optimizer, step_size: int, gamma: float=0.1) -> None:
        """
        Initialize the StepLR.
        
        Parameters:
        - optimizer: The optimizer for which to schedule the learning rate.
        - step_size (int): The number of epochs after which to decay the learning rate.
        - gamma (float, optional): The factor by which to multiply the learning rate when decaying.
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.current_iters = 0
    
    def step(self) -> None:
        """
        Call this method after each epoch. This will update the learning rate if necessary.
        """
        self.current_iters += 1
        if self.current_iters % self.step_size == 0:
            self.optimizer.lr *= self.gamma

class ConstantLR(object):
    def __init__(self, optimizer, factor: float=1./3, total_iters: int=5) -> None:
        """
        Initialize the ConstantLR.

        Parameters:
        - optimizer: The optimizer for which to schedule the learning rate.
        - factor (float, optional): The factor by which to multiply the learning rate when decaying.
        - total_iters (int, optional): The number of iterations after which to decay the learning rate.
        """
        self.optimizer = optimizer
        self.factor = factor
        self.total_iters = total_iters
        self.init_lr = optimizer.lr
        self.constant_lr = self.init_lr * self.factor
        self.current_iters = 0

    def step(self) -> None:
        """
        Call this method after each iteration. This will update the learning rate if necessary.
        """
        self.current_iters += 1
        if self.current_iters < self.total_iters:
            self.optimizer.lr = self.constant_lr
        elif self.current_iters == self.total_iters:
            self.optimizer.lr = self.init_lr

class MultiStepLR(object):
    def __init__(self, optimizer, milestones: list, gamma: float=0.1) -> None:
        """
        Initialize the MultiStepLR.

        Parameters:
        - optimizer: The optimizer for which to schedule the learning rate.
        - milestones (list): List of epoch indices. Must be increasing.
        - gamma (float, optional): The factor by which to multiply the learning rate when decaying.
        """
        self.optimizer = optimizer
        self.milestones = iter(milestones)
        self.current_milestone = next(self.milestones, float('inf'))
        self.gamma = gamma
        self.current_iters = 0
    
    def step(self) -> None:
        """
        Call this method after each iteration. This will update the learning rate if necessary.
        """
        self.current_iters += 1
        if self.current_iters == self.current_milestone:
            self.optimizer.lr *= self.gamma
            self.current_milestone = next(self.milestones, float('inf'))