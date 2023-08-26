# Multilayer Perceptron from Scratch using NumPy
<p align="center">
  <img src="./outcomes/MLP.jpg">
  <br>
  Multilayer Perceptron
</p>

## :sparkles: 1. Introduction
This repository contains a framework for a Multilayer Perceptron implemented solely using NumPy, with the exception of one imported SciPy function. This project draws inspiration from both TensorFlow and PyTorch. As you delve deeper, you'll notice that constructing a model with this framework bears a resemblance to the style of TensorFlow. Furthermore, some specific implementations within the source code are influenced by PyTorch. If you're keen on understanding the design and usage of this framework, let's dive in!

## :sparkles: 2. A topy example
Let's dive into a toy example.
First, we need to import the necessary packages.
```python
import numpy as np
import matplotlib.pyplot as plt
from mlperceptron import Dense, MultilayerPerceptron
```
Then, we need to generate some random data points. Please note that for reproductivity, a random seed should be set.
```python
np.random.seed(3407)
class_1 = np.hstack([np.random.normal( 1, 1, size=(500, 2)),  np.ones(shape=(500, 1))])
class_2 = np.hstack([np.random.normal(-1, 1, size=(200, 2)), -np.ones(shape=(200, 1))])
dataset = np.vstack([class_1, class_2])
X, y = dataset[:,:2], dataset[:,2]
```
Let's take a quick look at what they look like.
```python
plt.figure(figsize=(6, 6))
plt.scatter(class_1[:,0], class_1[:,1], label='1')
plt.scatter(class_2[:,0], class_2[:,1], label='-1')
plt.grid()
plt.legend()
plt.show()
```

<p align="center">
  <img src="./outcomes/random_points.png">
  <br>
  Random Data Points
</p>

It looks cool! Let's construct our model.
```python
layers = [
    Dense(2, 4, activation='leaky_relu', init='kaiming_normal', init_params={'mode': 'out'}),
    Dense(4, 3, activation='hardswish', init='xavier_normal'),
    Dense(3, 2, activation='relu', init='kaiming_normal', init_params={'mode': 'in'}),
    Dense(2, 1, activation='tanh', init='xavier_uniform')
]
mlp = MultilayerPerceptron(layers)
mlp.compile(optimizer='Adam',
            loss='MeanSquareError')
mlp.fit(X, y, epochs=80)
```
We've already done it. Easy? That's what we want!
Let's check its loss through epochs.
```python
loss = mlp.loss_tracker()
plt.figure(figsize=(15,4))
plt.plot(loss)
plt.grid()
```
<p align="center">
  <img src="./outcomes/toy_loss.png">
  <br>
  Loss through epochs
</p>

Not bad! It seems to work as expected.

Finally, let's look at the decision boundary of our model.
<p align="center">
  <img src="./outcomes/toy_decision_boundary.png">
  <br>
  Decision Boundary
</p>

**That's awesome!!**

## :sparkles: 3. Key modules
### Activations
- Hyperbolic Tangent(Tanh)
  - Formula:      
    $\mathrm{Tanh}(x) = \mathrm{tanh}(x) = \frac{\mathrm{exp}(x) - \mathrm{exp}(-x)}{\mathrm{exp}(x) + \mathrm{exp}(-x)}$
- Tanhshrink
  - Formula:      
    $\mathrm{Tanhshrink}(x) = x − \mathrm{tanh}(x)$
- Hardtanh
- Sigmoid
  - Formula:      
    $\mathrm{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \mathrm{exp}(-x)}$
- LogSigmoid
  - Formula:      
    $\mathrm{LogSigmoid}(x) = \mathrm{log}(\frac{1}{1 + \mathrm{exp}(-x)})$
- Hardsigmoid
- ReLU
  - Formula:
    $\mathrm{ReLU}(x) = \mathrm{max}(0, x)$
- ReLU6
  - Formula:
    $\mathrm{ReLU6}(x) = \mathrm{min}(\mathrm{max}(0,x),6)$
- leaky relu
  - Formula:      
    $\mathrm{LeakyReLU}(x) = \mathrm{max}(0,x) + \alpha * \mathrm{min}(0,x)$
- ELU
- CELU
  -  Formula:      
    $\mathrm{CELU}(x) = \mathrm{max}(0,x) + \mathrm{min}(0, \alpha * (\mathrm{exp}(x/\alpha) − 1))$
- SELU
  - Formula:      
    $\mathrm{SELU}(x) = \mathrm{scale} * (\mathrm{max}(0,x) + \mathrm{min}(0, \alpha * (\mathrm{exp}(x) − 1)))$
- GELU
  - Formula:      
    $\mathrm{GELU}(x) = 0.5 * x * (1 + \mathrm{Tanh}(\sqrt{\frac{2}{\pi}})) * (x + 0.044715 * x^3)$
- Mish
  - Formula:      
    $\mathrm{Mish}(x) = x * \mathrm{Tanh}(\mathrm{Softplus}(x))$
- Swish
  - Formula:      
    $\mathrm{Swish}(x)  = x * \sigma(x)$
- Hardswish
- Softplus
  - Formula:      
    $\mathrm{Softplus}(x)= \frac{1}{\beta} * \mathrm{log}(1 + \mathrm{exp}(\beta * x))$
- SoftSign
  - Formula:      
    $\mathrm{SoftSign}(x) = \frac{x}{1 + \vert x \vert}$
- SoftShrinkage
- HardShrink
- Threshold

Note that due to GitHub's markdown not supporting the piecewise function, some function formulas are not provided.

### Layers
- **Dense**
  - Dense Layer (Fully Connected Layer)
  - **Definition**:      
    A Dense Layer, also known as a Fully Connected Layer, is a layer in a neural network where every input node (or neuron) is connected to every output node. It's termed "fully connected" or "dense" because all inputs and outputs are interconnected.
  - **Mathematical Representation**:      
    $y = f(Wx + b)$
    
- **BatchNorm**
  - Batch Normalization
  - **Definition**:      
    Batch Normalization is a technique used in neural networks to standardize the activations of a given input layer on a mini-batch, which helps to stabilize and accelerate the training process.
  - **Mathematical Representation**:      
    For a given mini-batch, $B$, of size $m$, with activations $x$:      
    $\mu_B = \frac{1}{m}\Sigma_{i=1}^m x_i$      
    $\sigma_B^2 = \frac{1}{m}\Sigma_{i=1}^m (x_i - \mu_B)^2$      
    $\hat {x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$        
    $y_i = \gamma \hat {x_i} + \beta$
- **Dropout**:
  - **Definition**:      
    Dropout is a regularization technique used in neural networks where, during training, random subsets of neurons are "dropped out" (i.e., set to zero) at each iteration. This prevents the network from becoming overly reliant on any particular neuron and promotes a more generalized model.
- **Activ**:
  - Activation Layer
  - **Definition**:
    An Activation Layer in a neural network is a layer that applies a non-linear function to its inputs, transforming the data to introduce non-linearity into the model. This non-linearity allows the network to learn from error and make adjustments, which is essential for learning complex patterns. 

### Optimizers
- SGD(including Momentum and Nesterov)
  - Required Parameter: lr
  - Default Parameter: momentum=0.0, nesterov=False, weight_decay=0.0
- Adagrad
  - Required Parameter: None
  - Default Parameter: lr=1.0, weight_decay=0.0, epsilon=1e-10
- Adadelta
  - Required Parameter: None
  - Default Parameter: lr=1.0, rho=0.9, epsilon=1e-06, weight_decay=0.0
- Adam
  - Required Parameter: None
  - Default Parameter: lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-08, weight_decay=0.0

## :sparkles: 4. Instructions
### Regularization
In the previous toy example, the structure of the model is:
```python
layers = [
    Dense(2, 4, activation='leaky_relu', init='kaiming_normal', init_params={'mode': 'out'}),
    Dense(4, 3, activation='hardswish', init='xavier_normal'),
    Dense(3, 2, activation='relu', init='kaiming_normal', init_params={'mode': 'in'}),
    Dense(2, 1, activation='tanh', init='xavier_uniform')
]
```
Given that our training data is relatively simple, regularizations aren't strictly necessary. However, if the model starts overfitting, consider applying regularization techniques. In such a scenario, the layers should be constructed as follows:
```python
layers = [
    Dense(2, 4, init='kaiming_normal', init_params={'mode': 'out'}),
    BatchNorm(4),
    Activ('leaky_relu'),
    Dropout(dropout_rate=0.2),
    Dense(4, 3, init='xavier_normal'),
    BatchNorm(3),
    Activ('hardswish'),
    Dropout(dropout_rate=0.2),
    Dense(3, 2, init='kaiming_normal', init_params={'mode': 'in'}),
    BatchNorm(2),
    Activ('relu'),
    Dropout(dropout_rate=0.2),
    Dense(2, 1, activation='tanh', init='xavier_uniform')
]
```
When employing regularization, follow this order: Linear -> BatchNorm -> Activ -> Dropout.

<p align="center">
  <img src="./outcomes/reg_decision_boundary.png">
  <br>
  Decision Boundary After Regularization
</p>

### Optimizer
When training the model, the optimizer can be defined in the toy example as below:
```python
mlp.compile(optimizer='Adam',
            loss='MeanSquareError')
mlp.fit(X, y, epochs=80)
```
The default optimizer is utilized during the training process. However, it can also be customized:
- SGD
```python
optimizer = SGD(lr=1e-2)
```
- SGD with momentum:
```python
optimizer = SGD(lr=1e-2, momentum=0.9)
```
- Nesterov SGD:
```python
optimizer = SGD(lr=1e-2, momentum=0.9, nesterov=True)
```
After defining the optimizer, compile the model:
```python
mlp.compile(optimizer=optimizer,
            loss='MeanSquareError')
```
For using default settings, these two methods are equivalent:
```python
mlp.compile(optimizer='Adam',
            loss='MeanSquareError')
```
and
```python
mlp.compile(optimizer=Adam(),
            loss='MeanSquareError')
```
<p align="center">
  <img src="./outcomes/opt_loss.png">
  <br>
  Loss using SGD
</p>
