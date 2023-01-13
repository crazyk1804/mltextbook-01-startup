import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from ch03 import datasets

X_train_std, X_test_std, y_train, y_test, X_combined, y_combined = datasets()

weights, params = [], []
for c in np.arange(-5, 5):
	lr = LogisticRegression(C=10.**c, random_state=1, multi_class='ovr')
	lr.fit(X_train_std, y_train)
	weights.append(lr.coef_[1])
	params.append(10.**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal_length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
