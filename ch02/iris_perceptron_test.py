import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors, markers
from tabulate import tabulate

from oop_ml_api.chapter02 import Perceptron
from util import plot_decision_regions

if __name__ == '__main__':
	S = '/'.join(['https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data'])
	print('URL: {}'.format(S))

	df = pd.read_csv(S, header=None, encoding='utf-8')
	print('last five data of iris dataset')
	print(tabulate(df.tail(), headers='keys', tablefmt='psql'))

	# setosa, versicolor 레이블 구분
	y = df.iloc[:100, 4].values
	y = np.where(y == 'Iris-setosa', -1, 1)

	# 꽃받침 길이와 꽃잎 길이 추출
	X = df.iloc[:100, [0, 2]].values

	# 산점도 그리기
	plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
	plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
	plt.xlabel('꽃받침 [cm]')
	plt.ylabel('꽃잎 [cm]')
	plt.legend(loc='upper left')
	plt.show()

	# 퍼셉트론 API 테스트
	ppn = Perceptron(eta=0.1, n_iter=10, features=['sepal length[cm]', 'petal length[cm]'], verbose=True)
	ppn.fit(X, y)
	plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
	plt.xlabel('반복 횟수')
	plt.ylabel('업데이트 수')
	plt.show()

	# 산점도 그리기
	plot_decision_regions(X, y, classifier=ppn)
	plt.xlabel('꽃받침 [cm]')
	plt.ylabel('꽃입 [cm]')
	plt.legend(loc='upper left')
	plt.show()

