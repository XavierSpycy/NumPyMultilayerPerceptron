import numpy as np
from typing import Union, Dict
from nn.functional import Activation

class Input(object):
    """
    Input layer
    """
    def __init__(self, input_dim: int) -> None:
        """
        Store the input dimension.

        Parameters:
        - input_dim (int): The dimension of the input.

        Returns:
        - None
        """
        self.n_out = input_dim

class Dense(object):
    """
    Dense layer
    """
    def __init__(self, units: int,
                 activation: Union[str, None]=None, activation_params: Dict[str, float]={},
                 init: str='xavier_uniform', init_params: Dict[str, Union[float, str]]={}) -> None:
        """
        Initialize the Dense layer.

        Parameters:
        - units (int): The number of neurons in the layer.
        - activation (str): The name of the activation function.
        - activation_params (dict): The parameters of the activation function.
        - init (str): The name of the initialization method.
        - init_params (dict): The parameters of the initialization method.
        """
        self.n_in = None
        self.n_out = units
        self.activation_prev = None
        self.activation_name = activation
        self.activation_obj = Activation(activation, activation_params)
        self.activation_f = self.activation_obj.f
        self.activation_deriv = None
        self.set_init_params(init_params)
        self.initializer = init

    def init_weights(self) -> None:
        """
        Initialize the weights and biases, and the gradients of weights and biases.
        """
        self.params = {}
        self.grads = {}
        # Initialize the weights
        if self.activation_name == 'linear' or self.activation_name is None:
            self.init_method(self.activation_name, 'ones')
        else:
            self.init_method(self.activation_name, self.initializer)
        self.init_method(self.activation_name, self.initializer)
        # Initialize the biases
        self.params['b'] = np.zeros((self.n_out, ))
        # Initialize the gradients of weights
        self.grads['W'] = np.zeros(self.params['W'].shape)
        # Initialize the gradients of biases
        self.grads['b'] = np.zeros(self.params['b'].shape)

    def set_init_params(self, init_params: Dict[str, Union[float, str]]) -> None:
        """
        Set the parameters of the initialization method.

        Parameters:
        - init_params (dict): The parameters of the initialization method.
        """
        # Set the default parameters
        self.init_params = {}
        self.init_params.setdefault('a', 0.0)
        self.init_params.setdefault('b', 1.0)
        self.init_params.setdefault('mean', 0.0)
        self.init_params.setdefault('std', 1.0)
        self.init_params.setdefault('val', 0.0)
        self.init_params.setdefault('mode', 'in')
        
        if isinstance(init_params, dict):
            for param, value in init_params.items():
                self.init_params[param] = value
        
    def init_method(self, activation: str, init: str) -> None:
        """
        Define the initialization method of the weights.

        Parameters:
        - activation (str): The name of the activation function.
        - init (str): The name of the initialization method.
        """
        # Compute the gain
        if activation == 'tanh':
            gain = 5 / 3
        elif activation == 'relu':
            gain = np.sqrt(2)
        elif activation == 'leaky_relu':
            negative_slope = self.activation_obj.params['negative_slope']
            gain = np.sqrt(2 / (1 + np.square(negative_slope)))
        elif activation == 'selu':
            gain = 3 / 4
        else:
            gain = 1
        
        # Initialize the weights according to the initialization method in PyTorch official documentations
        if init == 'uniform':
            a, b = self.init_params['a'], self.init_params['b']
            self.params['W'] = gain * np.random.uniform(low=a, high=b, size=(self.n_in, self.n_out))
        elif init == 'normal':
            mean, std = self.init_params['mean'], self.init_params['std']
            self.params['W'] = gain * np.random.normal(loc=mean, scale=std, size=(self.n_in, self.n_out))
        elif init == 'constant':
            val = self.init_params['val']
            self.params['W'] = val * np.ones((self.n_in, self.n_out))
        elif init == 'zeros':
            self.params['W'] = np.zeros((self.n_in, self.n_out))
        elif init == 'ones':
            self.params['W'] = np.ones((self.n_in, self.n_out))
        elif init == 'xavier_uniform':
            a = np.sqrt(6. / (self.n_in + self.n_out))
            self.params['W'] = gain * np.random.uniform(low=-a, high=a, size=(self.n_in, self.n_out))
        elif init == 'xavier_normal':
            std = np.sqrt(2. / (self.n_in + self.n_out))
            self.params['W'] = gain * np.random.normal(loc=0.0, scale=std, size=(self.n_in, self.n_out))
        elif init == 'kaiming_uniform':
            mode = self.init_params['mode']
            if mode == 'in':
                bound = gain * np.sqrt(3. / self.n_in)
            elif mode == 'out':
                bound = gain * np.sqrt(3. / self.n_out)
            self.params['W'] = gain * np.random.uniform(low=-bound, high=bound, size=(self.n_in, self.n_out))
        elif init == 'kaiming_normal':
            mode = self.init_params['mode']
            if mode == 'in':
                std = gain * np.sqrt(self.n_in)
            elif mode == 'out':
                std = gain * np.sqrt(self.n_out)
            self.params['W'] = gain * np.random.normal(loc=0.0, scale=std, size=(self.n_in, self.n_out))
    
    def get_n_in(self, n_in: int) -> None:
        """
        Set the number of input neurons.

        Parameters:
        - n_in (int): The number of input neurons.
        """
        self.n_in = n_in
        self.init_weights()

    def get_activation_name(self) -> str:
        """
        Get the name of the activation function.

        Returns:
        - activation_name (str): The name of the activation function.
        """
        return self.activation_name
    
    def add_activation_deriv(self, activation_prev: str) -> None:
        """
        Add the derivative of the activation function.

        Parameters:
        - activation_prev (str): The name of the activation function of the previous layer.
        """
        self.activation_deriv = Activation(activation_prev).f_deriv
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        Parameters:
        - inputs (np.ndarray): The inputs of the layer.

        Returns:
        - outputs (np.ndarray): The outputs of the layer.
        """
        # Compute the linear outputs
        lin_outputs = np.dot(inputs, self.params['W']) + self.params['b']
        # Compute the activation outputs
        self.outputs = self.activation_f(lin_outputs)
        # Store the inputs
        self.inputs = inputs
        return self.outputs
    
    def backward(self, delta: np.ndarray, is_last_layer: bool=False) -> np.ndarray:
        """
        Backward propagation.

        Parameters:
        - delta (np.ndarray): The delta of the layer.
        - is_last_layer (bool): Whether the layer is the last layer.

        Returns:
        - delta (np.ndarray): The delta of the previous layer.
        """
        # Compute the gradients of weights and biases    
        self.grads['W'] = np.atleast_2d(self.inputs).T.dot(np.atleast_2d(delta))
        self.grads['b'] = np.sum(delta, axis=0)
        # Compute the delta of the previous layer
        delta = np.dot(delta, self.params['W'].T)
        # If the layer is not the last layer, compute the delta of the previous layer
        if not is_last_layer and self.activation_deriv is not None:
            delta *= self.activation_deriv(self.inputs)
        return delta

class Activ(object):
    """
    Activation layer
    """
    def __init__(self, activation: str='linear', activation_params: Dict={}) -> None:
        """
        Initialize the Activation layer.

        Parameters:
        - activation (str): The name of the activation function.
        - activation_params (dict): The parameters of the activation function.
        """
        self.n_in = None
        self.activation_prev = None
        self.activation_name = activation
        self.activation_obj = Activation(activation, activation_params)
        self.activation_f = self.activation_obj.f
        self.activation_deriv = None
        self.params = {}
    
    def get_n_in(self, n_in: int) -> None:
        """
        Set the number of input neurons.

        Parameters:
        - n_in (int): The number of input neurons.
        """
        self.n_in = n_in
        self.n_out = n_in
        self.params['W'] = np.ones((n_in, n_in))
        
    def get_activation_name(self) -> str:
        return self.activation_name
    
    def add_activation_deriv(self, activation_prev: str) -> None:
        self.activation_deriv = Activation(activation_prev).f_deriv
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        Parameters:
        - inputs (np.ndarray): The inputs of the layer.
        """
        self.outputs = self.activation_f(np.dot(inputs, self.params['W']))
        self.inputs = inputs
        return self.outputs
    
    def backward(self, delta: np.ndarray, is_last_layer: bool=False) -> np.ndarray:
        """
        Backward propagation.

        Parameters:
        - delta (np.ndarray): The delta of the layer.
        - is_last_layer (bool): Whether the layer is the last layer.

        Returns:
        - delta (np.ndarray): The delta of the previous layer.
        """ 
        # Compute the gradients of weights and biases    
        self.grads['W'] = np.atleast_2d(self.inputs).T.dot(np.atleast_2d(delta))
        self.grads['b'] = np.sum(delta, axis=0)
        # Compute the delta of the previous layer
        delta = np.dot(delta, self.params['W'].T)
        # If the layer is not the last layer, compute the delta of the previous layer
        if not is_last_layer and self.activation_deriv is not None:
            delta *= self.activation_deriv(self.inputs)
        return delta
    
class BatchNorm(object):
    """
    Batch normalization layer
    """
    def __init__(self, n_features: int, momentum: float=0.9, epsilon: float=1e-5) -> None:
        """
        Initialize the BatchNorm layer.

        Parameters:
        - n_features (int): The number of features.
        - momentum (float): The momentum of the moving average.
        - epsilon (float): The epsilon value.
        """
        self.n_features = self.n_in = self.n_out = n_features
        self.momentum = momentum
        self.epsilon = epsilon
        self.params = {}
        self.grads = {}
        self.params['gamma'] = np.ones(self.n_features)
        self.params['beta'] = np.zeros(self.n_features)
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self.grads['gamma'] = np.zeros(self.n_features)
        self.grads['beta'] = np.zeros(self.n_features)
        self.x_normalized = None
        self.xmu = None
        self.ivar = None
        self.training = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        Parameters:
        - inputs (np.ndarray): The inputs of the layer.

        Returns:
        - outputs (np.ndarray): The outputs of the layer.
        """
        # If the layer is in training mode, compute the outputs using batch normalization
        if self.training:
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            self.xmu = inputs - batch_mean
            self.ivar = 1. / np.sqrt(batch_var + self.epsilon)
            self.x_normalized = self.xmu * self.ivar
            out = self.params['gamma'] * self.x_normalized + self.params['beta']
            self.running_mean = self.momentum * self.running_mean + (1. - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1. - self.momentum) * batch_var
        # Otherwise, compute the outputs using the running mean and variance
        else:
            xmu = inputs - self.running_mean
            ivar = 1. / np.sqrt(self.running_var + self.epsilon)
            x_normalized = xmu * ivar
            out = self.params['gamma'] * x_normalized + self.params['beta']
        return out

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Backward propagation.

        Parameters:
        - delta (np.ndarray): The delta of the layer.
        """
        N, D = delta.shape
        # Compute the gradients of weights and biases
        self.grads['gamma'] = np.sum(delta * self.x_normalized, axis=0)
        self.grads['beta'] = np.sum(delta, axis=0)
        # Normalize the delta
        dx_normalized = delta * self.params['gamma']
        # Compute the delta of mean and variance
        dvar = np.sum(dx_normalized * self.xmu * -0.5 * np.power(self.ivar, 3), axis=0)
        dmean = np.sum(dx_normalized * -self.ivar, axis=0) + dvar * np.mean(-2. * self.xmu, axis=0)
        dx = dx_normalized * self.ivar + dvar * 2. * self.xmu / N + dmean / N
        return dx

class Dropout(object):
    """
    Dropout layer
    """
    def __init__(self, dropout_rate: float=0.5) -> None:
        """
        Initialize the Dropout layer.

        Parameters:
        - dropout_rate (float): The dropout rate.
        """
        self.dropout_rate = dropout_rate
        self.training = True
        self.mask = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        Parameters:
        - inputs (np.ndarray): The inputs of the layer.

        Returns:
        - outputs (np.ndarray): The outputs of the layer.
        """
        # If the layer is in training mode, compute the outputs using dropout mask
        if self.training:
            # Generate the dropout mask
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape) / (1 - self.dropout_rate)
            return inputs * self.mask # Multiply the inputs by the dropout mask
        # Otherwise, return the inputs
        else:
            return inputs

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Backward propagation.

        Parameters:
        - delta (np.ndarray): The delta of the layer.

        Returns:
        - delta (np.ndarray): The delta of the previous layer.
        """
        # If the layer is in training mode, compute the delta using dropout mask
        if self.training:
            return delta * self.mask
        # Otherwise, return the delta
        else:
            return delta