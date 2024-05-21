import numpy as np

def logitrv(p: np.array):
	return np.log(p/(1-p))


def logit(x: np.array):
	return np.exp(x) / (1 + np.exp(x))