import numpy as np
from sklearn.model_selection import train_test_split

from general import os_env, iris_dataset, standardized_scaler

# os_env()


def datasets():
	X, y = iris_dataset(use_sklearn=True)
	X = X[:, [2, 3]]
	sc = standardized_scaler(X, y)
	X_std = sc.transform(X)

	X_train_std, X_test_std, y_train, y_test = train_test_split(
		X_std, y, test_size=.3, random_state=1, stratify=y
	)
	X_combined = np.vstack((X_train_std, X_test_std))
	y_combined = np.hstack((y_train, y_test))

	return X_train_std, X_test_std, y_train, y_test, X_combined, y_combined
