import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from LogisticRegression import LogisticRegression


def calculate_accuracy(y_predicted, y_real):
    return np.sum(y_predicted == y_real) / len(y_real)


breast_cancer_dataset = datasets.load_breast_cancer()
X, y = breast_cancer_dataset.data, breast_cancer_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

logisticRegressionClassifier = LogisticRegression(learning_rate=0.1, number_of_iterations=1000)
logisticRegressionClassifier.fit(X_train, y_train)
y_pred = logisticRegressionClassifier.predict(X_test)

accuracy = calculate_accuracy(y_pred, y_test)
print(accuracy)
