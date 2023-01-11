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


class AdalineGD(object):
	"""
	적응형 선형 뉴런 분류기

	속성
	-----------------------------------------------------------
	w_
		: id-array 학습된 가중치
	const_
		: list 에포크마다 누적된 비용 함수의 제곱합
	"""

	def __init__(self, eta=0.01, n_iter=50, random_state=1):
		"""
		훈련 데이터 학습

		매게변수
		---------------------------------------------------------
		:param eta
			float 학습률(0.0과 1.0사이)
		:param n_iter
			int 훈련데이터 반복 횟수
		:param random_state
			int 가중치 무작위 초기화를 위한 나느수 생성기 시드
		"""
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state
		self.w_, self.cost_ = None, None

	def fit(self, X, y):
		"""
		훈련 데이터 학습

		매개변수
		----------------------------------------------------------
		:param X: { array-list }, shape = [n_samples, n_features]
			n_sample 개의 샘플과 n_features 개의 특성으로 이루어진 훈련데이터
		:param y: array-like, shape = [n_samples]
			타깃값
		:return self
		"""
		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			net_input = self.net_input(X)
			output = self.activation(net_input)
			errors = (y - output)
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] = self.eta * errors.sum()
			cost = (errors ** 2).sum() / 2.0
			self.cost_.append(cost)

		return self

	def net_input(self, X):
		""" 최종 입력 계산 """
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		""" 선형 활성화 계산 """
		# todo// 뭐 하는거 없는데?
		return X

	def predict(self, X):
		""" 단위 계단 함수를 사용하여 클래스 레이블을 반환 """
		return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class AdalineSGD(object):
	"""
	Adaptive Linear Neuron 분류기

	속성
	---------------------------------------------------------------
	w_
		id-array 학습된 가중치
	cost_
		list 모든 훈련 샘플에 대해 에포크마다 누적된 평균 비용 함수의 제곱합
	"""

	def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
		"""
		:param eta
			float 학습률
		:param n_iter
			int 훈련 데이터셋 반복 횟수
		:param shuffle
			True 로 설정하면 같은 반복이 되지 않도록 에포크마다 훈련 데이터를 섞는다
		:param random_state
			가중치 무작위 초기화를 위한 난수 생성기 시드
		"""
		self.eta = eta
		self.n_iter = n_iter
		self.shuffle = shuffle
		self.random_state = random_state

		self.w_initialized = False
		self.w_, self.cost_ = [], []

		self.rgen = np.random.RandomState(self.random_state)

	def fit(self, X, y):
		"""
		훈련 데이터 학습

		:param X
			{ array-like }, shape = [n_samples, n_features]
			n_sample 개의 샘플과 n_features 개의 특성으로 이루어진 훈련 데이터
		:param y
			array-like, shape = [n_samples]
		:return: AdalineSGD
		"""
		self._initialize_weights(X.shape[1])
		self.cost_ = []
		for i in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X, y)
			cost = []
			for xi, target in zip(X, y):
				cost.append(self._update_weights(xi, target))
			avg_cost = sum(cost) / len(cost)
			self.cost_.append(avg_cost)
		return self

	def partial_fit(self, X, y):
		"""
		가중치를 다시 초기화하지 않고 훈련 데이터를 학습

		:param X
			{ array-like }, shape = [n_samples, n_features]
			n_sample 개의 샘플과 n_features 개의 특성으로 이루어진 훈련 데이터
		:param y
			array-like, shape = [n_samples]
		:return: AdalineSGD
		"""
		if not self.w_initialized:
			self._initialize_weights(X.shape[1])
		if y.ravel().shape[0] > 1:
			for xi, target in zip(X, y):
				self._update_weights(xi, target)
		else:
			self._update_weights(X, y)

		return self

	def _shuffle(self, X, y):
		""" 훈련 데이터 섞기 """
		r = self.rgen.permutation(len(y))
		return X[r], y[r]

	def _initialize_weights(self, m):
		""" 랜덤한 작은 수로 가중치를 초기화 """
		self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
		self.w_initialized = True

	def _update_weights(self, xi, target):
		""" 아달린 학습 규칙을 적용하여 가중치를 업데이트 """
		output = self.activation(self.net_input(xi))
		error = target - output
		self.w_[1:] += self.eta * xi.dot(error)
		self.w_[0] += self.eta * error
		cost = 0.5 * error ** 2
		return cost

	def net_input(self, X):
		"""
		최종 입력 계산
		todo//
			이게 볼때마다 결과가 헷갈리는데
			dot 연산은 하나의 행으로 계산하면 값이 숫자로 나오지만
			여러개의 행으로 계산이 되면 각 계산 결과의 ndarray 가 리턴된다
		"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		""" 선형 활성화 계산 """
		return X

	def predict(self, X):
		""" 단위 계단 함수를 사용하여 클래스 레이블을 반환 """
		return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
