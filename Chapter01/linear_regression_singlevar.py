#%%
import sys

import numpy as np

#filename = sys.argv[1]
filename = 'data_singlevar.txt'
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

#%% Train/test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train = np.array(X[:num_training]).reshape((num_training, 1))
y_train = np.array(y[:num_training])
# X_train.shape => (40, 1), which is a matrix
# y_train.shape => (40,), which is an array
# type(X_train) => numpy.ndarray
# type(y_train) => numpy.ndarray

# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

# Create linear regression object
from sklearn import linear_model

linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

# Predict the output
y_test_pred = linear_regressor.predict(X_test)

# Plot outputs
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# Measure performance
import sklearn.metrics as sm

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

#In [27]: a1 = np.array([1,2,3,4,5])
#In [28]: b2 = np.array([5,4,3,2,1])
#In [31]: c2 = np.array([1,2,5,4,3])
#In [29]: sm.mean_squared_error(a1, b2)
#Out[29]: 8.0
#In [32]: sm.mean_squared_error(a1, c2)
#Out[32]: 1.6000000000000001
#In [38]: sm.mean_squared_error(a1, a1)
#Out[38]: 0.0
#In [30]: sm.explained_variance_score(a1, b2)
#Out[30]: -3.0
#In [37]: sm.explained_variance_score(a1, c2)
#Out[37]: 0.19999999999999996
#In [39]: sm.explained_variance_score(a1, a1)
#Out[39]: 1.0

# Model persistence
import cPickle as pickle

output_model_file = '3_model_linear_regr.pkl'

with open(output_model_file, 'w') as f:
    pickle.dump(linear_regressor, f)

with open(output_model_file, 'r') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
