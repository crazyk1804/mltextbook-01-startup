import math

from matplotlib import pyplot as plt
from sklearn.svm import SVC

from ch03 import datasets
from util import plot_decision_regions

X_train_std, X_test_std, y_train, y_test, X_combined, y_combined = datasets()
svm = SVC(kernel='linear', C=1, random_state=1)
svm.fit(X_train_std, y_train)
plot_decision_regions(
	X_combined, y_combined, classifier=svm, test_idx=range(105, 150)
)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
