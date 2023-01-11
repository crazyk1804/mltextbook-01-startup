from matplotlib import pyplot as plt

from ch02.OOMachineLearningAPI import AdalineSGD
from general import iris_dataset, standardize
from util import plot_decision_regions

X, y = iris_dataset()
X_std = standardize(X)

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardize]')
plt.ylabel('petal length [standardize]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.show()
