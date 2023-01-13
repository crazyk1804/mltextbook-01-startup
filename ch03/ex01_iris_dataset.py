from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from general import print_anl, iris_dataset
from util import plot_decision_regions

X, y = iris_dataset(use_sklearn=True)
X = X[:, [2, 3]]
print('클래스 레이블: ', np.unique(y))


def print_splitted_data_count(result):
	X_train, X_test, y_train, y_test = result
	print('X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(
		len(X_train), len(X_test), len(y_train), len(y_test)
	))
	print('y 레이블 카운트: ', np.bincount(y))
	print('y_train 레이블 카운트: ', np.bincount(y_train))
	print('y_test 레이블 카운트: ', np.bincount(y_test))

	return result


print('\n테스트 셋 분리')
print_splitted_data_count(
	train_test_split(
		X, y, test_size=.3, random_state=1
	)
)

print('\nstratify 는 계층화 기능 여부')
X_train, X_test, y_train, y_test = print_splitted_data_count(
	train_test_split(
		X, y, test_size=.3, random_state=1, stratify=y
	)
)

# 표준화 처리
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 퍼셉트론 훈련
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print_anl('잘못 분류된 샘플 개수: {}'.format((y_test != y_pred).sum()))
print('정확도: {:.3f}'.format(accuracy_score(y_test, y_pred)))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(
	X=X_combined_std, y=y_combined,
	test_idx=range(105, 150),
	# X=X_train_std, y=y_train,
	classifier=ppn,
)
plt.xlabel('petal length [standardize]')
plt.ylabel('petal width [standardize]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
