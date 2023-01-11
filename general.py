import os
import platform

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager


def os_env():
	plt.rcParams['axes.unicode_minus'] = False
	if platform.system() == 'Linux':
		# 한글 폰트가 설치되어 있어야 함
		plt.rcParams['font.family'] = 'NanumGothic'
	elif platform.system() == 'Windows':
		# 한글이 깨지는 문제와 관련한 내용으로 vmoptions 뒤에
		# -Dfile.encoding=UTF-8
		# -Dconsole.encoding=UTF-8
		# 2가지를 같이 붙여줘야 한다
		os.system('chcp 65001')

		font_path = 'C:/Windows/Fonts/NGULIM.TTF'
		font = font_manager.FontProperties(fname=font_path).get_name()
		plt.rcParams['font.family'] = font


def iris_dataset():
	S = '/'.join(['https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data'])
	print('URL: {}'.format(S))

	df = pd.read_csv(S, header=None, encoding='utf-8')
	# print('last five data of iris dataset')
	# print(tabulate(df.tail(), headers='keys', tablefmt='psql'))

	# setosa, versicolor 레이블 구분
	y = df.iloc[:100, 4].values
	y = np.where(y == 'Iris-setosa', -1, 1)

	# 꽃받침 길이와 꽃잎 길이 추출
	X = df.iloc[:100, [0, 2]].values

	return X, y


def standardize(X):
	X_std = np.copy(X)
	for feature_idx in range(X.shape[1]):
		feature = X_std[:, feature_idx]
		X_std[:, feature_idx] = (feature - feature.mean()) / feature.std()

	return X_std


def print_anl(message):
	print('\n{}'.format(message))


def sigmoid(z):
	# np.exp => 자연상수 e 의 제곱수를 구하는 함수
	# ex) np.exp(3) = e**3 = 2.718281828459045 ** 3 = 20.085536923187664
	return 1.0 / (1.0 + np.exp(-z))
