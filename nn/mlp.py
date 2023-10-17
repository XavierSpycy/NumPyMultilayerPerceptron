import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from typing import Generator
from nn.layers import Dropout, BatchNorm
from nn.functional import Activation
from nn.optim import Optimizer

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

    def compile(self, optimizer, loss, scheduler=None):
        if isinstance(optimizer, str):
            self.optimizer = Optimizer(optimizer)
        else:
            self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler if scheduler is not None else None
        
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
            y_hat_clipped = np.clip(y_hat, 1e-10, 1-1e-10)
            loss = -np.sum(y * np.log(y_hat_clipped)) / y.shape[0]
            delta = y_hat_clipped - y
        return loss, delta
    
    def one_hot_encode(self, y):
        """
        Helper function to one-hot encode labels.
        """
        y = y.astype(int)
        unique_labels = np.array(np.unique(y)).tolist()
        self.n_classes = len(np.unique(y))
        
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        
        one_hot = np.zeros((y.shape[0], self.n_classes))
        encoded_indices = [self.label_to_index[label] for label in y]
        one_hot[np.arange(y.shape[0]), encoded_indices] = 1
        return one_hot
    
    def training_time(self):
        return self.tracker[0]
    
    def loss_tracker(self):
        return self.tracker[1]
    
    def backward(self, delta):
        delta = self.layers[-1].backward(delta, output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    def fit(self, X, y, batch_size:int=32, epochs:int=100, callbacks=[], use_progress_bar:bool=False):
        def conditional_tqdm(iterable: range, use_progress_bar: bool=False) -> Generator[int, None, None]:
            """
            Determine whether to use tqdm or not based on the use_progress_bar flag.

            Parameters:
            - iterable (range): Range of values to iterate over.
            - use_progress_bar (bool, optional): Whether to print progress bar. Default is False.

            Returns:
            - item (int): Current iteration.
            """

            if use_progress_bar:
                from tqdm import tqdm
                for item in tqdm(iterable):
                    yield item
            else:
                for item in iterable:
                    yield item

        X = np.array(X)
        flag = False
        
        if self.loss == 'CrossEntropy':
            y = self.one_hot_encode(y)
        else:
            y = np.array(y).reshape(-1, 1)

        loss_tracker = np.zeros(epochs)
        self.training = True
        
        start_time = time.time()
        for k in conditional_tqdm(range(epochs), use_progress_bar):
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
            if self.scheduler is not None:
                self.scheduler.step()
            if len(callbacks) > 0:
                for callback in callbacks:
                    flag = callback.callback(self)
                    if flag:
                        break
            if flag:
                break
        end_time = time.time()
        self.tracker = [end_time - start_time, loss_tracker, ]
        
    def predict(self, x, predict_probs=False):
        self.training = False
        x = np.array(x)
        output = self.forward(x)
        if self.loss == 'CrossEntropy':
            if not predict_probs:
                class_indices = np.argmax(output, axis=1)
                decoded_labels = [self.index_to_label[idx] for idx in class_indices]
                return np.array(decoded_labels).reshape(-1)
        else:
            return output.reshape(-1)
    
    def save(self, filename):
        joblib.dump(self, filename)
        
    @staticmethod
    def load(filename: str) -> 'MultilayerPerceptron':
        """
        Load a saved model.

        Parameters:
        - filename (str): Name of the file to load.

        Returns:
        - model (MultilayerPerceptron): The saved model.
        """
        return joblib.load(filename)
    
    def plot_loss(self, figsize:tuple=(15, 4), grid: bool=True, xlabel: str='Epochs', ylabel: str='Loss') -> None:
        loss = self.loss_tracker()
        plt.figure(figsize=figsize)
        plt.plot(loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if grid:
            plt.grid()
        plt.show()

    def get_weights(self):
        """
        Return the weights of the model.

        Returns:
        - weights (dict): A dictionary containing the weights of the model.
        """

        weights = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for key, value in layer.params.items():
                    weights[f"layer_{idx}_{key}"] = value
        return weights
    
    def set_weights(self, weights):
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for key in layer.params.keys():
                    weight_key = f"layer_{idx}_{key}"
                    if weight_key in weights:
                        layer.params[key] = weights[weight_key]