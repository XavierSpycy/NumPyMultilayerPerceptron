import numpy as np
from datasets.data_loader import datasets
from nn.mlp import MultilayerPerceptron
from nn.layers import Dense, Dropout
from nn.optim import Adam
from nn.schedules import MultiStepLR
from sklearn.metrics import accuracy_score as accuracy

# Load the data
X_train, y_train = datasets(train=True)
X_test, y_test = datasets(train=False)

# Set the random seed
np.random.seed(3407)

# Build the model
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
optimizer = Adam(lr=1e-3, weight_decay=0.02)
scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.8)
mlp.compile(optimizer=optimizer,
            loss='CrossEntropy',
            scheduler=scheduler)
# Train the model
mlp.fit(X_train, y_train, epochs=500, batch_size=32, use_progress_bar=True)
# Evaluate the model
loss = mlp.loss_tracker()
train_time = mlp.training_time()
print(f'Training time: {train_time:.2f} second(s).')
print(f'Loss: {loss[-1]:.2f}.')
mlp.plot_loss()

print(f"Accuracy on the training set is: {accuracy(y_train, mlp.predict(X_train)):.2%}." )
print(f"Accuracy on the test set is: {accuracy(y_test, mlp.predict(X_test)):.2%}.")

# mlp.save('mlp.pickle')