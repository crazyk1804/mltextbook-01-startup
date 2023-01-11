import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02, test_idx=None):
	# 마커와 컬러맵 설정
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# 결정경계 그리기
	# 차트 영역을 잘게 쪼개서 모든 범위의 좌표를 생성한 후
	# 그 좌표들에 대해 예측을 처리하고 결과를 다른 색상으로 그리는 방식인 듯 하다
	# resolution 을 올렸을때 예상처럼 경계에 aliasing 이 생기지만
	# 직각이 아닌 사선으로 나오는건 잘 모르겠다
	# contourf 함수에 대해 내용을 봐야 할 것 같다
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(
		np.arange(x1_min, x1_max, resolution),
		np.arange(x2_min, x2_max, resolution)
	)
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	# 산점도 그리기
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(
			x=X[y == cl, 0],
			y=X[y == cl, 1],
			alpha=0.8,
			c=colors[idx],
			marker=markers[idx],
			label=cl,
			# edgecolors='black'
		)

	# chapter 03 에서 추가
	# 테스트 샘플을 부각하여 그린다
	if test_idx:
		X_test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(
			X_test[:, 0], X_test[:, 1],
			facecolors='none',
			edgecolor='black',
			alpha=1.0,
			linewidth=1,
			marker='o',
			s=100,
			label='test set'
		)
