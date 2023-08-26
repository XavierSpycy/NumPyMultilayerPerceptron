import time
import numpy as np
from scipy.optimize import fsolve

class Activation(object):
    def inverse(self, f, y):
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
        return np.where(x > 0.0, x, alpha * (np.exp(x) - 1))
    
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
    
    def __init__(self, activation, **kwargs):
        self.params = {}
        self.params.setdefault('min_val', -1.0) # HardTanh
        self.params.setdefault('max_val', 1.0) # HardTanh
        self.params.setdefault('negative_slope', 1e-2) # Leaky ReLU
        self.params.setdefault('alpha', 1.0) # ELU & CELU
        self.params.setdefault('beta', 1.0) # Softplus
        self.params.setdefault('lambd', 0.5) # SoftShrinkage # HardShrinkage
        self.params.setdefault('threshold', 0.0) # Threshold
        self.params.setdefault('value', 0.0) # Threshold
        
        if isinstance(kwargs, dict):
            for param, value in kwargs.items():
                self.params[param]  = value
            
        if activation == 'linear':
            self.f = self.linear
            self.f_deriv = self.linear_deriv
        elif activation == 'sigmoid':
            self.f = self.sigmoid
            self.f_deriv = self.sigmoid_deriv
        elif activation == 'logsigmoid':
            self.f = self.logsigmoid
            self.f_deriv = self.logsigmoid_deriv
        elif activation == 'hardsigmoid':
            self.f = self.hardsigmoid
            self.f_deriv = self.hardsigmoid_deriv
        elif activation == 'tanh':
            self.f = self.tanh
            self.f_deriv = self.tanh_deriv
        elif activation == 'tanhshrink':
            self.f = self.tanhshrink
            self.f_deriv = self.tanhshrink_deriv
        elif activation == 'hardtanh':
            self.f = self.hardtanh
            self.f_deriv = self.hardtanh_deriv
        elif activation == 'relu':
            self.f = self.relu
            self.f_deriv = self.relu_deriv
        elif activation == 'relu6':
            self.f = self.relu6
            self.f_deriv = self.relu6_deriv
        elif activation == 'leaky_relu':
            self.f = self.leaky_relu
            self.f_deriv = self.leaky_relu_deriv
        elif activation == 'elu':
            self.f = self.elu
            self.f_deriv = self.elu_deriv
        elif activation == 'selu':
            self.f = self.selu
            self.f_deriv = self.selu_deriv
        elif activation == 'celu':
            self.f = self.celu
            self.f_deriv = self.celu_deriv
        elif activation == 'gelu':
            self.f = self.gelu
            self.f_deriv = self.gelu_deriv
        elif activation == 'mish':
            self.f = self.mish
            self.f_deriv = self.mish_deriv
        elif activation == 'swish':
            self.f = self.swish
            self.f_deriv = self.swish_deriv
        elif activation == 'hardswish':
            self.f = self.hardswish
            self.f_deriv = self.hardswish_deriv
        elif activation == 'softplus':
            self.f = self.softplus
            self.f_deriv = self.softplus_deriv
        elif activation == 'softsign':
            self.f = self.softsign
            self.f_deriv = self.softsign_deriv
        elif activation == 'softshrink':
            self.f = self.softshrink
            self.f_deriv = self.softshrink_deriv
        elif activation == 'hardshrink':
            self.f = self.hardshrink
            self.f_deriv = self.hardshrink_deriv
        elif activation == 'threshold':
            self.f = self.threshold
            self.f_deriv = self.threshold_deriv
        elif activation == 'softmax':
            self.f = self.softmax

class Dense(object):    
    def __init__(self, n_in=None, n_out=None,
                 activation='linear', activation_params={},
                 init=None, init_params={}):
        self.n_in = n_in
        if n_out is None:
            n_out = n_in
        self.n_out = n_out
        self.params = {}
        self.grads = {}
        self.activation_prev = None
        self.activation_name = activation
        self.activation_obj = Activation(activation, **activation_params)
        self.activation_f = self.activation_obj.f
        self.activation_deriv = None
        
        self.set_init_params(init_params)
        if init is None:
            self.init(activation, 'ones')
        else:
            self.init(activation, init)
        self.params['b'] = np.zeros((n_out, ))
        self.grads['W'] = np.zeros(self.params['W'].shape)
        self.grads['b'] = np.zeros(self.params['b'].shape)
        
    def set_init_params(self, init_params):
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
        
    def init(self, activation, init):
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
        
    def get_activation_name(self):
        return self.activation_name
    
    def add_activation_deriv(self, activation_prev):
        self.activation_deriv = Activation(activation_prev).f_deriv
    
    def forward(self, inputs):
        lin_outputs = np.dot(inputs, self.params['W']) + self.params['b']
        self.outputs = self.activation_f(lin_outputs)
        self.inputs = inputs
        return self.outputs
    
    def backward(self, delta, output_layer=False):         
        self.grads['W'] = np.atleast_2d(self.inputs).T.dot(np.atleast_2d(delta))
        self.grads['b'] = np.sum(delta, axis=0)
        if self.activation_deriv is not None:
            delta = delta.dot(self.params['W'].T) * self.activation_deriv(self.inputs)
        return delta

class Activ(object):    
    def __init__(self, activation='linear', activation_params={}):
        self.n_in = None
        self.activation_prev = None
        self.activation_name = activation
        self.activation_obj = Activation(activation, **activation_params)
        self.activation_f = self.activation_obj.f
        self.activation_deriv = None
        self.params = {}
    
    def get_n_in(self, n_in):
        self.n_in = n_in
        self.n_out = n_in
        self.params['W'] = np.ones((n_in, n_in))
        
    def get_activation_name(self):
        return self.activation_name
    
    def add_activation_deriv(self, activation_prev):
        self.activation_deriv = Activation(activation_prev).f_deriv
    
    def forward(self, inputs):
        self.outputs = self.activation_f(np.dot(inputs, self.params['W']))
        self.inputs = inputs
        return self.outputs
    
    def backward(self, delta):         
        if self.activation_deriv is not None:
            delta = delta.dot(self.params['W'].T) * self.activation_deriv(self.inputs)
        return delta
    
class BatchNorm(object):
    def __init__(self, n_features, momentum=0.9, epsilon=1e-5):
        self.n_features = n_features
        self.n_in = self.n_out = n_features
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

    def forward(self, inputs):
        if self.training:
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            self.xmu = inputs - batch_mean
            self.ivar = 1. / np.sqrt(batch_var + self.epsilon)
            self.x_normalized = self.xmu * self.ivar
            out = self.params['gamma'] * self.x_normalized + self.params['beta']
            self.running_mean = self.momentum * self.running_mean + (1. - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1. - self.momentum) * batch_var
        else:
            xmu = inputs - self.running_mean
            ivar = 1. / np.sqrt(self.running_var + self.epsilon)
            x_normalized = xmu * ivar
            out = self.params['gamma'] * x_normalized + self.params['beta']
        return out

    def backward(self, delta):
        N, D = delta.shape
        self.grads['gamma'] = np.sum(delta * self.x_normalized, axis=0)
        self.grads['beta'] = np.sum(delta, axis=0)
        dx_normalized = delta * self.params['gamma']
        dvar = np.sum(dx_normalized * self.xmu * -0.5 * np.power(self.ivar, 3), axis=0)
        dmean = np.sum(dx_normalized * -self.ivar, axis=0) + dvar * np.mean(-2. * self.xmu, axis=0)
        dx = dx_normalized * self.ivar + dvar * 2. * self.xmu / N + dmean / N
        return dx

class Dropout(object):
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.training = True
        self.mask = None
        
    def forward(self, inputs):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape) / (1 - self.dropout_rate)
            return inputs * self.mask
        else:
            return inputs

    def backward(self, delta): 
        if self.training:
            return delta * self.mask
        else:
            return delta

class MultilayerPerceptron(object):
    def __init__(self, layers):
        self.layers = layers
        num_layers = len(layers)
        i = 0
        while i < num_layers - 1:
            if isinstance(self.layers[i], Dropout):
                i += 1
                continue
            if self.layers[i].n_in is None:
                self.layers[i].get_n_in(n_out)
            if hasattr(self.layers[i], 'get_activation_name'):
                activation_name = self.layers[i].get_activation_name()
                j = i + 1
                while j < num_layers and not hasattr(self.layers[j], 'add_activation_deriv'):
                    j += 1
                if j < num_layers:
                    self.layers[j].add_activation_deriv(activation_name)
            n_out = self.layers[i].n_out
            i += 1
        self.output_activation = layers[-1].get_activation_name()

    def compile(self, optimizer, loss):
        if isinstance(optimizer, str):
            self.optimizer = Optimizer(optimizer)
        else:
            self.optimizer = optimizer
        self.loss = loss
        
    def forward(self, inputs):
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNorm)):
                layer.training = self.training
            outputs = layer.forward(inputs)
            inputs = outputs
        return outputs

    def criterion(self, y, y_hat):
        if self.loss == 'MeanSquareError':
            activation_deriv = Activation(self.output_activation).f_deriv
            error = y - y_hat
            loss = np.mean(error ** 2)
            delta = -(error * activation_deriv(y_hat))
        elif self.loss == 'CrossEntropy':
            loss = -np.sum(y * np.log(y_hat + 1e-9)) / y.shape[0]
            delta = y_hat - y
        return loss, delta
    
    def training_time(self):
        return self.tracker[0]
    
    def loss_tracker(self):
        return self.tracker[1]
    
    def backward(self,delta):
        delta = self.layers[-1].backward(delta, output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    def fit(self,X, y, batch_size=8, epochs=100):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        loss_tracker = np.zeros(epochs)
        self.training = True
        
        start_time = time.time()
        for k in range(epochs):
            loss = np.zeros(X.shape[0] // batch_size)
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            batchs = np.array_split(indices, X.shape[0] // batch_size)
            it = 0
            for batch in batchs:
                X_batch = X[batch]
                y_batch = y[batch]
                y_hat = self.forward(X_batch)
                loss[it], delta = self.criterion(y_batch, y_hat)
                self.backward(delta)
                self.optimizer.update(self.layers)
                it += 1
            loss_tracker[k] = np.mean(loss)
        end_time = time.time()
        self.tracker = [end_time - start_time, loss_tracker, ]
        
        
    def predict(self, x):
        self.training = False
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i,:])
        return output

class Optimizer(object):
    def __init__(self, name):
        if name == 'SGD':
            self.optimizer = SGD(0.1)
        elif name == 'Adagrad':
            self.optimizer = Adagrad()
        elif name == 'Adadelta':
            self.optimizer = Adadelta()
        elif name == 'Adam':
            self.optimizer = Adam()
    
    def update(self, layers):
        self.optimizer.update(layers)

class SGD(object):
    def __init__(self, lr, momentum=0.0, nesterov=False, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocity = None
        if self.nesterov:
            self.velocity_prev = {}
            
    def update(self, layers):
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
                    
    def init_velocity(self, layers):
        self.velocity = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.velocity[i] = {}
                for key in layer.params:
                    self.velocity[i][key] = np.zeros_like(layer.params[key])

class Adagrad(object):
    def __init__(self, lr=1.0, weight_decay=0.01, epsilon=1e-10):
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
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-08, weight_decay=0.0):
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