import numpy as np

class Optimizer(object):
    """
    An optimizer selector.
    """
    def __init__(self, name: str) -> None:
        """
        Initialize the optimizer selector.

        Parameters:
        - name (str): The name of the optimizer.
        """

        if name == 'SGD':
            self.optimizer = SGD(0.1)
        elif name == 'Adagrad':
            self.optimizer = Adagrad()
        elif name == 'Adadelta':
            self.optimizer = Adadelta()
        elif name == 'Adam':
            self.optimizer = Adam()
    
    def update(self, layers:list) -> None:
        """
        Define the update rule for the parameters of the model.

        Parameters:
        - layers (list): A list of layers in the model.
        """

        self.optimizer.update(layers)

class SGD(object):
    """
    SGD optimizer, including SGD with momentum and Nesterov accelerated gradient.
    """
    def __init__(self, lr, momentum: float=0.0, nesterov: bool=False, weight_decay: float=0.0) -> None:
        """
        Initialize the SGD optimizer.

        Parameters:
        - lr (float): Learning rate.
        - momentum (float, optional): Momentum factor. Default is 0.0.
        - nesterov (bool, optional): Whether to use Nesterov accelerated gradient (NAG). Default is False.
        - weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.0.
        """

        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocity = None
        if self.nesterov:
            self.velocity_prev = {}
            
    def update(self, layers: list) -> None:
        """
        Update the parameters of the model.

        Parameters:
        - layers (list): A list of layers in the model.
        """
        if self.velocity is None:
            self.init_velocity(layers)
            
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    grad = layer.grads[key]
                    grad += self.weight_decay * layer.params[key]
                    if self.nesterov:
                        self.velocity_prev[key] = self.velocity[i][key]
                        self.velocity[i][key] = self.momentum * self.velocity_prev[key] - self.lr * grad
                        layer.params[key] -= -self.momentum * self.velocity_prev[key] - (1 + self.momentum) * self.velocity[i][key]
                    else:
                        self.velocity[i][key] = self.momentum * self.velocity[i][key] + self.lr * grad
                        layer.params[key] -= self.velocity[i][key]
                    
    def init_velocity(self, layers: list) -> None:
        """
        Initialize the velocity.

        Parameters:
        - layers (list): A list of layers in the model.
        """

        self.velocity = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.velocity[i] = {}
                for key in layer.params:
                    self.velocity[i][key] = np.zeros_like(layer.params[key])

class Adagrad(object):
    def __init__(self, lr=1.0, weight_decay=0.0, epsilon=1e-10):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.grad_square = None
        
    def update(self, layers):
        if self.grad_square is None:
            self.init_grad_square(layers)
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    grad = layer.grads[key]
                    grad += self.weight_decay * layer.params[key]
                    self.grad_square[i][key] += np.square(grad)
                    layer.params[key] -= self.lr * grad / (np.sqrt(self.grad_square[i][key]) + self.epsilon)
    
    def init_grad_square(self, layers):
        self.grad_square = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.grad_square[i] = {}
                for key in layer.params:
                    self.grad_square[i][key] = np.zeros_like(layer.params[key])

class Adadelta(object):
    def __init__(self, lr=1.0, rho=0.9, epsilon=1e-06, weight_decay=0.0):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.accum_grad_square = None
        self.accum_delta_square = None
    
    def update(self, layers):
        if self.accum_grad_square is None:
            self.init_accum_square(layers)
    
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    grad = layer.grads[key]
                    grad += self.weight_decay * layer.params[key]
                    self.accum_grad_square[i][key] *= self.rho
                    self.accum_grad_square[i][key] += (1 - self.rho) * np.square(layer.grads[key])
                    delta = - np.sqrt(self.accum_delta_square[i][key] + self.epsilon) / np.sqrt(self.accum_grad_square[i][key] + self.epsilon) * layer.grads[key]
                    layer.params[key] += self.lr * delta
                    self.accum_delta_square[i][key] *= self.rho
                    self.accum_delta_square[i][key] += (1 - self.rho) * np.square(delta)
                    
    def init_accum_square(self, layers):
        self.accum_grad_square = {}
        self.accum_delta_square = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.accum_grad_square[i] = {}
                self.accum_delta_square[i] = {}
                for key in layer.params:
                    self.accum_grad_square[i][key] = np.zeros_like(layer.params[key])
                    self.accum_delta_square[i][key] = np.zeros_like(layer.params[key])

class Adam(object):
    """
    Adam optimizer.
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-08, weight_decay=0.0) -> None:
        """
        Initialize the Adam optimizer.

        Parameters:
        - lr (float, optional): Learning rate. Default is 1e-3.
        - beta1 (float, optional): Exponential decay rate for the first moment estimates. Default is 0.9.
        - beta2 (float, optional): Exponential decay rate for the second moment estimates. Default is 0.999.
        - epsilon (float, optional): A small constant for numerical stability. Default is 1e-8.
        - weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.0.
        """

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.first_moment = None
        self.second_moment = None
        self.t = 1
        
    def update(self, layers):
        if self.first_moment is None:
            self.init_moment(layers)
            
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    grad = layer.grads[key]
                    grad += self.weight_decay * layer.params[key]
                    self.first_moment[i][key] *= self.beta1
                    self.second_moment[i][key] *= self.beta2
                    self.first_moment[i][key] += (1 - self.beta1) * layer.grads[key]
                    self.second_moment[i][key] += (1 - self.beta2) * np.square(layer.grads[key])
                    first_moment_hat = self.first_moment[i][key] / (1 - self.beta1 ** self.t)
                    second_moment_hat = self.second_moment[i][key] / (1 - self.beta2 ** self.t)
                    layer.params[key] -= self.lr * first_moment_hat / (np.sqrt(second_moment_hat) + self.epsilon)
        self.t += 1
        
    def init_moment(self, layers):
        self.first_moment = {}
        self.second_moment = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.first_moment[i] = {}
                self.second_moment[i] = {}
                for key in layer.params:
                    self.first_moment[i][key] = np.zeros_like(layer.params[key])
                    self.second_moment[i][key] = np.zeros_like(layer.params[key])