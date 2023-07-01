import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression


def mean_squared_error(y_test, predictions):
    return np.mean((y_test - predictions) ** 2)


X, y = datasets.make_regression(n_samples=200, n_features=1, noise=20, random_state=16)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

regression_model = LinearRegression(learning_rate=0.2)
regression_model.fit(X_train, y_train)
predictions = regression_model.predict(X_test)

mean_squared_error = mean_squared_error(y_test, predictions)
print(mean_squared_error)

y_pred_line = regression_model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
