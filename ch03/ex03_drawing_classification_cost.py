import numpy as np
from matplotlib import pyplot as plt

from general import sigmoid, os_env

os_env()

def cost_1(z):
	return -np.log(sigmoid(z))


def cost_0(z):
	return -np.log(1 - sigmoid(z))

if __name__ == '__main__':
	z = np.arange(-10, 10, 0.1)
	phi_z = sigmoid(z)

	c1 = [cost_1(x) for x in z]
	plt.plot(phi_z, c1, label='J(w) y=1일때')
	c0 = [cost_0(x) for x in z]
	plt.plot(phi_z, c0, label='J(w) y=0일때')

	plt.ylim(0.0, 5.1)
	plt.xlim(0, 1)
	plt.xlabel('$phi$(z)')
	plt.ylabel('J(w)')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.show()
