from datasets.data_loader import datasets
from model.mlperceptron import MultilayerPerceptron
from eval.metrics import accuracy

X_train, y_train = datasets(train=True)
X_test, y_test = datasets(train=False)

mlp = MultilayerPerceptron.load('model_hub/mlp.pickle')

print(f"Accuracy on the training set is: {accuracy(y_train, mlp.predict(X_train)):.2%}." )
print(f"Accuracy on the test set is: {accuracy(y_test, mlp.predict(X_test)):.2%}.")