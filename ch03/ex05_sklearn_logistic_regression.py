from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from ch03 import datasets
from util import plot_decision_regions

X_train_std, X_test_std, y_train, y_test, X_combined, y_combined = datasets()

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
plot_decision_regions(
	X_combined, y_combined, classifier=lr, test_idx=range(105, 150)
)
plt.xlabel('pteal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
