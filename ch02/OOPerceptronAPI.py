import numpy as np
import pandas as pd
from tabulate import tabulate


class Perceptron(object):
	"""
	퍼셉트론 분류기
	todo// 이렇게 todo// 표시를 하는건 내가 직접 작성하는 주석으로 해야할 일이 아님

	속성
	-----------------------------------------------------------
	: w_
		id-array	학습된 가중치
	: errors_
		list		에포크마다 누적된 분류 오류
	"""

	def __init__(self, eta=0.01, n_iter=50, random_state=1, verbose=False, features=None):
		"""
		매개변수
		-----------------------------------------------------------
		:param eta
			float	학습률(0.0 에서 1.0 사이)
		:param n_iter
			int	훈련 데이터셋 반복 횟수
		:param random_state
			int	가중치 무작위 초기화를 위한 난수 생성기 시드
		:param verbose
			bool 과정 출력 여부
		"""
		self.eta, self.n_iter, self.random_state = eta, n_iter, random_state
		self.w_, self.errors_ = 0, []
		self.verbose, self.verbose_prefix = verbose, ''
		self.features = features

	def fit(self, X, y):
		"""
		훈련 데이터 학습

		매개변수
		:param X
			{array-like}, shape = [n_samples, n_features] n_samples 개의 샘플과 n_features 개의 특성으로 이루어진 훈련 데이터
		:param y
			array-like, shape = [n_samples] 타깃값(레이블)
		:return:
		"""
		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
		self.errors_ = []
		df = pd.DataFrame(
			columns=
				['iter', 'target', 'predict', 'update', 'w', 'w-update', 'errors'] +
				[self.features[i] if len(self.features) > i else i for i in range(len(self.features))]
		)
		df_idx = 0

		self.say('initial intercept: {}'.format(self.w_[0]))
		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				predict = self.predict(xi).tolist()
				update = self.eta * (target - predict)
				df_data = [_ + 1, target, predict, update, self.w_[0]]
				self.w_[1:] += update * xi
				# todo// 0번째 에는 가중치 변화량 저장
				df_data += [self.w_[0] + update]
				self.w_[0] += update
				if abs(update) > 0.0:
					errors += int(update != 0.0)
					df_data += [errors] + xi.tolist()
					df.loc[df_idx] = df_data
				df_idx += 1
			# todo// 어느만큼의 에러가 발생했는지 iter 별로 확인 가능 하게끔 속성값에 추가
			self.errors_.append(errors)

		self.say(tabulate(df, headers='keys', tablefmt='psql'))

	def say(self, words):
		if self.verbose:
			print(words)

	def net_input(self, X):
		""" 입력 계산 """
		# todo// 그동안 쌓인 가중치 변화량을 절편으로 사용하는게 이해 안됨
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, X):
		""" 단위 계단 함수를 사용하여 클래스 레이블을 반환 """
		return np.where(self.net_input(X) >= 0.0, 1, -1)
