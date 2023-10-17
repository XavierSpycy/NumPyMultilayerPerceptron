from datasets.data_loader import datasets
from nn.mlp import MultilayerPerceptron
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score

# Load the data
X_train, y_train = datasets(train=True)
X_test, y_test = datasets(train=False)

# Load the model
mlp = MultilayerPerceptron.load('model_hub/mlp.pickle')
weights = mlp.get_weights()
# print(weights)

# Evaluate the model
print(f"Accuracy on the training set is: {accuracy(y_train, mlp.predict(X_train)):.2%}." )
print(f"Accuracy on the test set is: {accuracy(y_test, mlp.predict(X_test)):.2%}.")
print(f"Precision on the training set is: {precision_score(y_train, mlp.predict(X_train), average='weighted'):.2%}." )
print(f"Precision on the test set is: {precision_score(y_test, mlp.predict(X_test), average='weighted'):.2%}.")