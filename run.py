import numpy as np
from datasets.data_loader import datasets
from nn.mlp import MultilayerPerceptron
from nn.layers import Input, Dense, Dropout
from nn.optim import Adam
from nn.schedules import MultiStepLR
from nn.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score as accuracy

# Load the data
X_train, y_train = datasets(train=True)
X_test, y_test = datasets(train=False)

# Set the random seed
np.random.seed(3407)

# Build the model
layers = [
    Input(input_dim=128),
    Dense(units=120, activation='elu', init='kaiming_uniform'),
    Dropout(dropout_rate=0.30),
    Dense(units=112,  init='kaiming_uniform'),
    Dropout(dropout_rate=0.25),
    Dense(units=96, activation='elu', init='kaiming_uniform'),
    Dropout(dropout_rate=0.20),
    Dense(units=64, activation='elu', init='kaiming_uniform'),
    Dropout(dropout_rate=0.15),
    Dense(units=48, activation='elu', init='kaiming_uniform'),
    Dropout(dropout_rate=0.10),
    Dense(units=32, activation='elu', init='kaiming_uniform'),
    Dense(units=24, activation='elu', init='kaiming_uniform'),
    Dense(units=16, activation='elu', init='kaiming_uniform'),
    Dense(units=10, activation='softmax')
]

mlp = MultilayerPerceptron(layers)
optimizer = Adam(lr=1e-3, weight_decay=0.2)
scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.8)
earlystopping = EarlyStopping(accuracy, patience=10, mode='max', restore_best_weights=True, start_from_epoch=20)
mlp.compile(optimizer=optimizer,
            metrics=['CrossEntropy', 'Accuracy'],
            scheduler=scheduler
)
# Train the model
mlp.fit(X_train, y_train, 
        epochs=90, batch_size=128, 
        validation_data=(X_test, y_test), use_progress_bar=True, 
        callbacks=[earlystopping]
)
# Evaluate the model
loss = mlp.loss_tracker()
train_time = mlp.training_time()
print(f'Training time: {train_time:.2f} second(s).')
print(f'Loss: {loss[-1]:.2f}.')
mlp.plot_loss()

print(f"Accuracy on the training set is: {accuracy(y_train, mlp.predict(X_train)):.2%}." )
print(f"Accuracy on the test set is: {accuracy(y_test, mlp.predict(X_test)):.2%}.")

# mlp.save('model_hub/mlp.pickle')