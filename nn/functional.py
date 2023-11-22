import numpy as np
from scipy.optimize import fsolve

from typing import Callable
class Activation(object):
    """
    Activation functions and their derivatives.
    """

    def inverse(self, f: Callable[[np.array], np.array], y: np.array) -> np.array:
        """
        Compute the inverse of an activation function.

        Parameters:
        f: Callable[[np.array], np.array]. The activation function.
        y: np.array. The output of the activation function.

        Returns:
        np.array. The input of the activation function.
        """
        if isinstance(y, np.ndarray):
            return np.array([self.inverse(f, y_val) for y_val in y])
        else:
            func = lambda x: float(f(float(x)) - y)
            return fsolve(func, 0.0)[0]
    
    def linear(self, x):
        return x
    
    def linear_deriv(self, a):
        return 1.0
    
    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, a):
        return 1.0 - a ** 2
    
    def tanhshrink(self, x):
        return x - np.tanh(x)
    
    def tanhshrink_deriv(self, a):
        return 1.0 - self.tanh_deriv(a)
    
    def hardtanh(self, x):
        min_val, max_val = self.params['min_val'], self.params['max_val']
        conditions = [x > max_val, x > min_val, (x < max_val) & (x > min_val)]
        choices = [max_val, min_val, x]
        return np.select(conditions, choices)
        
    def hardtanh_deriv(self, a):
        return np.where((a == -1.0) | (a == 1.0), 0.0, 1.0)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_deriv(self, a):
        return a * (1.0 - a)
    
    def logsigmoid(self, x):
        return np.log(self.sigmoid(x))
    
    def logsigmoid_deriv(self, a):
        x = self.inverse(self.logsigmoid, a)
        return 1.0 - self.sigmoid(x)
    
    def hardsigmoid(self, x):
        conditions = [x >= 3.0, x <= -3.0, (x < 3.0) & (x > -3.0)]
        choices = [1.0, 0.0, x/6 + 1/2]
        return np.select(conditions, choices)
    
    def hardsigmoid_deriv(self, a):
        return np.where((a == 1.0) | (a == 0.0), 0.0, 1/6)
        
    def relu(self, x):
        return np.maximum(x, 0.0)
    
    def relu_deriv(self, a):
        return np.where(a > 0.0, 1.0, 0.0)
    
    def relu6(self, x):
        return np.minimum(np.maximum(x, 0.0), 6.0)
    
    def relu6_deriv(self, a):
        return np.where((a < 6.0) & (a > 0.0), 1.0, 0.0)
    
    def leaky_relu(self, x):
        negative_slope = self.params['negative_slope']
        return np.where(x > 0.0, x, negative_slope * x)
    
    def leaky_relu_deriv(self, a):
        negative_slope = self.params['negative_slope']
        return np.where(a > 0.0, 1.0, negative_slope)
        
    def elu(self, x):
        alpha = self.params['alpha']
        x_clipped = np.clip(x, a_min=-np.log(np.finfo(x.dtype).max / alpha), a_max=None)
        out = np.where(x_clipped > 0.0, x_clipped, alpha * (np.exp(x_clipped) - 1))
        return out
    
    def elu_deriv(self, a):
        alpha = self.params['alpha']
        return np.where(a > 0.0, 1.0, a + alpha)
    
    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return np.where(x > 0.0, scale * x, alpha * (np.exp(x / alpha) - 1.0))
    
    def selu_deriv(self, a):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        x = np.where(a > 0.0, a / scale, alpha * np.log(a / alpha + 1.0))
        return np.where(x > 0.0, scale, alpha * np.exp(x / alpha))
    
    def celu(self, x):
        alpha = self.params['alpha']
        return np.where(x > 0.0, x, alpha * (np.exp(x / alpha) - 1.0))
    
    def celu_deriv(self, a):
        alpha = self.params['alpha']
        x = np.where(a > 0.0, a, alpha * np.log(a / alpha + 1.0))
        return np.where(x > 0.0, 1.0, alpha * np.exp(x / alpha))
    
    def gelu(self, x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def gelu_deriv(self, a):
        x = self.inverse(self.gelu, a)
        return (0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
                + 0.5 * x 
                * (1.0 - np.square(np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))) 
                * np.sqrt(2 / np.pi) * (1.0 + 0.134145 * np.square(x)))
    
    def mish(self, x):
        return x * np.tanh(self.softplus(x))
    
    def mish_deriv(self, a):
        x = self.inverse(self.mish, a)
        temp = np.tanh(np.log(1 + np.exp(x)))
        return temp + x * (1 - np.square(temp)) * self.sigmoid(x)
    
    def swish(self, x):
        return x * self.sigmoid(x)
    
    def swish_deriv(self, a):
        x = self.inverse(self.swish, a)
        return self.sigmoid(x) + x * self.sigmoid(x) * (1.0 - self.sigmoid(x))
    
    def hardswish(self, x):
        conditions = [x <= -3.0, x >= 3.0, (x < 3.0) & (x > -3.0)]
        choices = [0, x, x * (x + 3.0) / 6.0]
        return np.select(conditions, choices)
    
    def hardswish_deriv(self, a):
        x = self.inverse(self.hardswish, a)
        return (2.0 * x + 3.0) / 6.0
        
    def softplus(self, x):
        beta = self.params['beta']
        return 1.0 / beta * np.log(1 + np.exp(beta * x))
    
    def softplus_deriv(self, a):
        beta = self.params['beta']
        x = self.inverse(self.softplus, a)
        return 1.0 / (1.0 + np.exp(-x * beta))
        
    def softsign(self, x):
        return x / (1.0 + np.abs(x))
    
    def softsign_deriv(self, a):
        x = self.inverse(self.softsign, a)
        return 1.0 / np.square(1.0 + np.abs(x))
    
    def softshrink(self, x):
        lambd = self.params['lambd']
        conditions = [x > lambd, x < lambd, x == lambd]
        choices = [x - lambd, x + lambd, 0.0]
        return np.select(conditions, choices)
    
    def softshrink_deriv(self, a):
        return np.where(a != 0, 1.0, 0.0)
    
    def hardshrink(self, x):
        lambd = self.params['lambd']
        return np.where((x < -lambd) | (x > lambd), x, 0.0)
    
    def hardshrink_deriv(self, a):
        return np.where(a != 0, 1.0, 0.0)
    
    def threshold(self, x):
        threshold = self.params['threshold']
        value = self.params['value']
        return np.where(x > threshold, x, value)
    
    def threshold_deriv(self, a):
        return np.where(a > 0.0, 1.0, 0.0)
    
    def softmax(self, x):
        x -= np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    def __init__(self, activation, activation_params={}):
        self.params = {}
        self.params.setdefault('min_val', -1.0) # HardTanh
        self.params.setdefault('max_val', 1.0) # HardTanh
        self.params.setdefault('negative_slope', 1e-2) # Leaky ReLU
        self.params.setdefault('alpha', 1.0) # ELU & CELU
        self.params.setdefault('beta', 1.0) # Softplus
        self.params.setdefault('lambd', 0.5) # SoftShrinkage # HardShrinkage
        self.params.setdefault('threshold', 0.0) # Threshold
        self.params.setdefault('value', 0.0) # Threshold
        self.f_deriv = None

        if isinstance(activation_params, dict):
            for param, value in activation_params.items():
                self.params[param]  = value
        activation = 'linear' if activation is None else activation
        self.f = getattr(self, activation)
        if activation != 'softmax':
            self.f_deriv = getattr(self, activation + '_deriv')