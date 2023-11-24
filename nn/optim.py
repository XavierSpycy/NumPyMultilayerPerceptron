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
        Update rule of SGD for the parameters of the model.

        Parameters:
        - layers (list): A list of layers in the model.
        """
        if self.velocity is None:
            self.init_velocity(layers)
            
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    # Get the gradient
                    grad = layer.grads[key]
                    # Add weight decay
                    grad += self.weight_decay * layer.params[key]
                    # If SGD with Nesterov accelerated gradient (NAG) is used
                    if self.nesterov:
                        # Update the velocity
                        self.velocity_prev[key] = self.velocity[i][key]
                        self.velocity[i][key] = self.momentum * self.velocity_prev[key] - self.lr * grad
                        # Update the parameters
                        layer.params[key] -= -self.momentum * self.velocity_prev[key] - (1 + self.momentum) * self.velocity[i][key]
                    else:
                        # Update the velocity
                        self.velocity[i][key] = self.momentum * self.velocity[i][key] + self.lr * grad
                        # Update the parameters
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
    """
    Adagrad optimizer.
    """
    def __init__(self, lr=1.0, weight_decay=0.0, epsilon=1e-10) -> None:
        """
        Initialize the Adagrad optimizer.

        Parameters:
        - lr (float, optional): Learning rate. Default is 1.0.
        - weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.0.
        - epsilon (float, optional): A small constant for numerical stability. Default is 1e-10.
        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.grad_square = None
        
    def update(self, layers: list) -> None:
        """
        Update rule of Adagrad for the parameters of the model.

        Parameters:
        - layers (list): A list of layers in the model.
        """
        # Initialize the grad_square
        if self.grad_square is None:
            self.init_grad_square(layers)
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    # Get the gradient
                    grad = layer.grads[key]
                    # Add weight decay
                    grad += self.weight_decay * layer.params[key]
                    # Update the grad_square
                    self.grad_square[i][key] += np.square(grad)
                    # Update the parameters
                    layer.params[key] -= self.lr * grad / (np.sqrt(self.grad_square[i][key]) + self.epsilon)
    
    def init_grad_square(self, layers: list) -> None:
        """
        Initialize the grad_square.

        Parameters:
        - layers (list): A list of layers in the model.
        """
        self.grad_square = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.grad_square[i] = {}
                for key in layer.params:
                    self.grad_square[i][key] = np.zeros_like(layer.params[key])

class Adadelta(object):
    """
    Adadelta optimizer.
    """
    def __init__(self, lr: float=1.0, rho: float=0.9, epsilon: float=1e-06, weight_decay: float=0.0):
        """
        Initialize the Adadelta optimizer.
        
        Parameters:
        - lr (float, optional): Learning rate. Default is 1.0.
        - rho (float, optional): Decay rate. Default is 0.9.
        - epsilon (float, optional): A small constant for numerical stability. Default is 1e-6.
        - weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.0.
        """
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.accum_grad_square = None
        self.accum_delta_square = None
    
    def update(self, layers: list) -> None:
        """
        Update rule of Adadelta for the parameters of the model.

        Parameters:
        - layers (list): A list of layers in the model.
        """
        if self.accum_grad_square is None:
            self.init_accum_square(layers)
    
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    # Get the gradient
                    grad = layer.grads[key]
                    # Add weight decay
                    grad += self.weight_decay * layer.params[key]
                    # Update the accum_grad_square
                    self.accum_grad_square[i][key] *= self.rho
                    self.accum_grad_square[i][key] += (1 - self.rho) * np.square(layer.grads[key])
                    # Calculate the delta
                    delta = - np.sqrt(self.accum_delta_square[i][key] + self.epsilon) / np.sqrt(self.accum_grad_square[i][key] + self.epsilon) * layer.grads[key]
                    # Update the accum_delta_square
                    layer.params[key] += self.lr * delta
                    # Update the accum_delta_square
                    self.accum_delta_square[i][key] *= self.rho
                    self.accum_delta_square[i][key] += (1 - self.rho) * np.square(delta)
                    
    def init_accum_square(self, layers: list) -> None:
        """
        Initialize the accum_grad_square and accum_delta_square.

        Parameters:
        - layers (list): A list of layers in the model.
        """
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
        
    def update(self, layers: list) -> None:
        """
        Update rule of Adam for the parameters of the model.

        Parameters:
        - layers (list): A list of layers in the model.
        """
        if self.first_moment is None:
            self.init_moment(layers)
            
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    # Get the gradient
                    grad = layer.grads[key]
                    # Add weight decay
                    grad += self.weight_decay * layer.params[key]
                    # Update biased first moment estimate
                    self.first_moment[i][key] *= self.beta1
                    # Update biased second raw moment estimate
                    self.second_moment[i][key] *= self.beta2
                    # Correct bias
                    self.first_moment[i][key] += (1 - self.beta1) * layer.grads[key]
                    self.second_moment[i][key] += (1 - self.beta2) * np.square(layer.grads[key])
                    # Update parameters
                    first_moment_hat = self.first_moment[i][key] / (1 - self.beta1 ** self.t)
                    second_moment_hat = self.second_moment[i][key] / (1 - self.beta2 ** self.t)
                    layer.params[key] -= self.lr * first_moment_hat / (np.sqrt(second_moment_hat) + self.epsilon)
        self.t += 1
        
    def init_moment(self, layers: list) -> None:
        """
        Initialize the first_moment and second_moment.

        Parameters:
        - layers (list): A list of layers in the model.
        """
        self.first_moment = {}
        self.second_moment = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.first_moment[i] = {}
                self.second_moment[i] = {}
                for key in layer.params:
                    self.first_moment[i][key] = np.zeros_like(layer.params[key])
                    self.second_moment[i][key] = np.zeros_like(layer.params[key])