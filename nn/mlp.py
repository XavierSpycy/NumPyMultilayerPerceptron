import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Generator
from nn.layers import Input, Dense, Dropout, BatchNorm, Activ
from nn.functional import Activation
from nn.optim import Optimizer
from nn.callbacks import EarlyStopping

class MultilayerPerceptron(object):
    def __init__(self, layers=[]):
        self.layers = layers
        self.loss = None
        self.accuracy = False
        self.stacked = False

    def stack_layers(self, layers):
        num_layers = len(layers)
        i = 0
        while i < num_layers:
            if isinstance(self.layers[i], Dropout):
                i += 1
                continue
            elif isinstance(self.layers[i], Input):
                n_out = self.layers[i].n_out
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
        self.layers[-1].n_in = n_out
        self.stacked = True

    def compile(self, optimizer, metrics, scheduler=None):
        if isinstance(optimizer, str):
            self.optimizer = Optimizer(optimizer)
        else:
            self.optimizer = optimizer
        for metric in metrics:
            if metric == 'MeanSquareError':
                self.loss = 'MeanSquareError'
            elif metric == 'CrossEntropy':
                self.loss = 'CrossEntropy'
            elif metric == 'Accuracy':
                self.accuracy = True
        self.scheduler = scheduler if scheduler is not None else None
        if not self.stacked:
            self.stack_layers(self.layers)
        
    def forward(self, inputs):
        for layer in self.layers:
            if isinstance(layer, Input):
                continue
            elif isinstance(layer, (Dropout, BatchNorm)):
                layer.training = self.training
            outputs = layer.forward(inputs)
            inputs = outputs
        return outputs
    
    def backward(self, delta):
        delta = self.layers[-1].backward(delta, is_last_layer=True)
        for layer in reversed(self.layers[:-1]):
            if isinstance(layer, Input):
                continue
            delta = layer.backward(delta)

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
    
    def one_hot_encode(self, y_):
        """
        Helper function to one-hot encode labels.

        Parameters:
        - y (np.array): The labels to encode.

        Returns:
        - one_hot (np.array): The one-hot encoded labels.
        """
        y = y_.copy()
        # Convert to integer type.
        y = y.astype(int)
        # Create a list of unique labels.
        unique_labels = np.array(np.unique(y)).tolist()
        # Get the number of classes.
        self.n_classes = len(np.unique(y))
        # Create a dictionary mapping each label to its index.
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        # Create a dictionary mapping each index to its label.
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        # Create a one-hot encoded matrix.
        one_hot = np.zeros((y.shape[0], self.n_classes))
        # Encode the labels.
        encoded_indices = [self.label_to_index[label] for label in y]
        one_hot[np.arange(y.shape[0]), encoded_indices] = 1
        return one_hot
    
    def training_time(self):
        return self.tracker['training_time']
    
    def loss_tracker(self):
        return self.tracker['loss']
    
    def accuracy_tracker(self):
        return self.tracker['accuracy']
    
    def save_weights(self):
        """
        Save the weights of the model.
        """
        return [layer.get_weights() for layer in self.layers if hasattr(layer, 'get_weights')]
    
    def load_weights(self, weights):
        """
        Load the weights of the model.
        """
        for layer, weights in zip(self.layers, weights):
            if hasattr(layer, 'set_weights'):
                layer.set_weights(weights)
    
    def fit(self, X_, y_, batch_size:int=32, epochs:int=100, validation_data=None, callbacks=[], use_progress_bar:bool=False):
        X = X_.copy()
        y = y_.copy()
        flag = False
        best_weights = None
        if validation_data is not None:
            X_val, y_val = validation_data

        if self.loss == 'CrossEntropy':
            y = self.one_hot_encode(y)
        else:
            y = np.array(y).reshape(-1, 1)

        loss_tracker = []
        accuracy_tracker = []
        
        start_time = time.time()
        for k in self.conditional_tqdm(range(epochs), use_progress_bar):
            self.training = True
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
            loss_tracker.append(np.mean(loss))
            if self.accuracy:
                y_pred = self(X, predict_probs=False)
                accuracy = np.mean(y_ == y_pred)
                accuracy_tracker.append(accuracy)
            if self.scheduler is not None:
                self.scheduler.step()
            if len(callbacks) > 0:
                for callback in callbacks:
                    if isinstance(callback, EarlyStopping):
                        y_val_pred = self(X_val, predict_probs=False)
                        flag, save_best_weights = callback(y_val, y_val_pred)
                        if save_best_weights:
                            best_weights = self.save_weights()
            if flag:
                break
        if best_weights is not None:
            self.load_weights(best_weights)
        end_time = time.time()
        self.tracker = {'training_time': end_time - start_time, 
                        'loss': loss_tracker, 
                        'accuracy': accuracy_tracker}
        
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

    def plot_accuracy(self, figsize:tuple=(15, 4), grid: bool=True, xlabel: str='Epochs', ylabel: str='Accuracy') -> None:
        accuracy = self.accuracy_tracker()
        plt.figure(figsize=figsize)
        plt.plot(accuracy)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
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
    
    def add(self, layer):
        """
        Add a layer to the model.

        Parameters:
        - layer (Layer): The layer to add to the model.
        """
        self.layers.append(layer)

    def __call__(self, x, predict_probs=False):
        """
        Override the __call__ method.

        Parameters:
        - x (np.ndarray): The input data.
        - predict_probs (bool, optional): Whether to return the predicted probabilities. Default is False.

        Returns:
        - output (np.ndarray): The output of the model.
        """
        return self.predict(x, predict_probs)
    
    def conditional_tqdm(self, iterable: range, use_progress_bar: bool=False) -> Generator[int, None, None]:
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