import numpy as np
import matplotlib.pyplot as plt
from datasets.data_loader import datasets
from model.mlperceptron import MultilayerPerceptron, Dense, Dropout, Adam
from eval.metrics import accuracy

X_train, y_train = datasets(train=True)
X_test, y_test = datasets(train=False)

np.random.seed(3407)

layers = [
    Dense(128, 120, activation='elu', init='kaiming_uniform'),
    Dropout(dropout_rate=0.25),
    Dense(120, 112, activation='elu', init='kaiming_uniform'),
    Dropout(dropout_rate=0.20),
    Dense(112, 96, activation='elu', init='kaiming_uniform'),
    Dropout(dropout_rate=0.15),
    Dense(96, 64, activation='elu', init='kaiming_uniform'),
    Dropout(dropout_rate=0.10),
    Dense(64, 48, activation='elu', init='kaiming_uniform'),
    Dropout(dropout_rate=0.05),
    Dense(48, 32, activation='elu', init='kaiming_uniform'),
    Dense(32, 24, activation='elu', init='kaiming_uniform'),
    Dense(24, 16, activation='elu', init='kaiming_uniform'),
    Dense(16, 10, activation='softmax', init='xavier_uniform')
]

mlp = MultilayerPerceptron(layers)
mlp.compile(optimizer=Adam(lr=1e-3, weight_decay=0.02),
            loss='CrossEntropy')
mlp.fit(X_train, y_train, epochs=500, batch_size=32)
loss = mlp.loss_tracker()
train_time = mlp.training_time()
print(f'Training time: {train_time:.2f} second(s).')
print(f'Loss: {loss[-1]:.2f}.')
plt.figure(figsize=(15, 4))
plt.plot(loss)
plt.grid()
plt.show()

print(f"Accuracy on the training set is: {accuracy(y_train, mlp.predict(X_train)):.2%}." )
print(f"Accuracy on the test set is: {accuracy(y_test, mlp.predict(X_test)):.2%}.")

# mlp.save('mlp.pickle')