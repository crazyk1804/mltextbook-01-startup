from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from general import iris_dataset, standardized_scaler
from oop_ml_api.chapter03 import LogisticRegressionGD
from util import plot_decision_regions

X, y = iris_dataset(use_sklearn=True)
X = X[:, [2, 3]]

sc = standardized_scaler(X, y)
X = sc.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=.3, random_state=1, stratify=y
)

X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
