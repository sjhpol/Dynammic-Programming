import numpy as np


def chkchain(pmtx):
	"""
	Checks to make sure Markov chain is suitable for manipulation
		Inputs:
			pmtx:       kxk transition matrix
						rows denote time t values, cols time t+1 values
		Output:
			isgood:     scalar (1=good,0=error)
	"""
	isgood = 1
	s = pmtx.shape[0]

	if s != pmtx.shape[1]:
		print("Error: Transition Matrix Not Square")
		isgood = 0
		return isgood

	tots = np.sum(pmtx, axis=1)
	if not np.allclose(tots, np.ones(s)):
		print("Error: Transition Probabilities Don't Sum to 1")
		isgood = 0
		return isgood

	return isgood


def chktval(pmtx, tval):
	"""
	Checks to make sure vector of values fits Markov chain
		Inputs:
			pmtx:       kxk transition matrix
						rows denote time t values, cols time t+1 values
			tval:       kx1 vector of values for variable
		Output:
			isgood:     scalar (1=good,0=error)
	"""
	isgood = 1
	s = pmtx.shape[0]

	if tval.shape[0] != s:
		print("Error: Transition Matrix Not Conforming to State Space")
		isgood = 0

	if np.allclose(tval, np.full((s, 1), tval[0])):
		print("Error: All States Identical")
		isgood = 0

	return isgood


def fdchain(pmtxl, tvall):
	"""
	Converts Markov chain in levels to chain in first differences
		Inputs:
			pmtxl:      kxk transition matrix
						rows denote time t values, cols time t+1 values
			tvall:      kx1 vector of values for variable
		Output:
			pmtxd:      (k^2)x(k^2) transition matrix of first differences
			tvald:      kx1 vector of first differences
	"""
	s = pmtxl.shape[0]

	tvald = (np.exp(-tvall) * np.exp(tvall).reshape(-1, 1)).transpose()
	tvald = np.log(np.ravel(tvald))
	pmtxd = np.zeros((s ** 2, s ** 2))

	isgood = chkchain(pmtxl)
	if isgood == 0:
		return None, None

	isgood = chktval(pmtxl, tvall)
	if isgood == 0:
		return None, None

	for i in range(s):
		shft = i * s
		for j in range(s):
			k = i + j * s
			pmtxd[k, shft: (i + 1) * s] = pmtxl[i, :]

	return pmtxd, tvald


def mrkinvar(pmtx):
	"""
	Finds the invariant matrix associated with a Markov Chain
		Inputs:
			pmtx:       kxk transition matrix
						rows denote time t values, cols time t+1 values
		Output:
			invar:      1xk vector of invariant transition probabilities
						(real portion)
	"""
	s = pmtx.shape[0]
	invar = -np.ones((s, 1))

	isgood = chkchain(pmtx)
	if isgood == 0:
		return None

	var, vec = np.linalg.eig(pmtx.T)
	va = np.diag(var)
	ve = vec

	va = np.isclose(va, np.ones((s, 1)))

	if np.sum(va) > 1:
		print("Error: Multiple Invariant Measures -- Can't Help You")
		return None

	# A = (np.eye(s) - pmtx.T) @ np.ones((1, s))
	A = np.concatenate([np.eye(s) - pmtx.T, np.ones((1, s))])
	A = np.linalg.inv(A.T @ A) @ A.T
	invar = np.real(A[:, -1])

	return invar


def mrkchpow(pmtx, power):
	"""
	Raises a Markov Chain to a Matrix power
		Inputs:
			pmtx:       kxk transition matrix
						rows denote time t values, cols time t+1 values
			power:      power to which matrix is raised
		Output:
			chntopow:   (real portion of) pmtx raised to power
	"""
	s = pmtx.shape[0]
	chntopow = -np.ones((s, s))

	isgood = chkchain(pmtx)
	if not isgood:
		return chntopow

	var, vec = np.linalg.eig(pmtx.T)
	va = np.diag(var)
	va = np.linalg.matrix_power(va, power)
	chntopow = np.dot(np.dot(vec, va), np.linalg.inv(vec))
	chntopow = np.real(chntopow.T)

	return chntopow


def mrkchev(pmtx, tval):
	"""
	Finds conditional and unconditional expectations of Markov Chain
		Inputs:
			pmtx:       kxk transition matrix
						rows denote time t values, cols time t+1 values
			tval:       kx1 vector of values for variable
		Output:
			condexp:    kx1 vector of conditional expectations
			ucondexp:   scalar of unconditional expectation
	"""
	s = pmtx.shape[0]
	condexp = -1e7 * np.ones((s, 1))
	ucondexp = -1e7

	isgood = chkchain(pmtx)
	if not isgood:
		return condexp, ucondexp

	isgood = chktval(pmtx, tval)
	if not isgood:
		return condexp, ucondexp

	invar = mrkinvar(pmtx)
	if np.allclose(invar, -np.ones(s)):
		return condexp, ucondexp

	condexp = pmtx @ tval
	ucondexp = invar @ tval

	return condexp, ucondexp


def mrkchvar(pmtx, tval):
	"""
	Finds conditional and unconditional variance of Markov Chain
		Inputs:
			pmtx:       kxk transition matrix
						rows denote time t values, cols time t+1 values
			tval:       kx1 vector of values for variable
		Output:
			condvar:    kx1 vector of conditional variances
			ucondvar:   scalar of unconditional variance
	"""
	s = pmtx.shape[0]
	# condexp = np.zeros((s, 1))
	# ucondexp = 0
	condvar = -np.ones((s, 1))
	ucondvar = -1

	isgood = chkchain(pmtx)
	if not isgood:
		return condvar, ucondvar

	isgood = chktval(pmtx, tval)
	if not isgood:
		return condvar, ucondvar

	invar = mrkinvar(pmtx)
	if np.allclose(invar, -np.ones(s)):
		return condvar, ucondvar

	condexp = pmtx @ tval
	ucondexp = invar @ tval

	# Conditional Deviations
	condev = (np.ones(s) * tval.reshape(-1,1)).transpose() - np.outer(condexp.flatten(), np.ones(s))
	condev = condev.transpose()
	condevsq = condev ** 2

	condvar = np.sum(pmtx * condevsq.T, axis=1).reshape(-1, 1)
	condvar = np.real(condvar)

	ucondev = tval.reshape(-1, 1) - ucondexp * np.ones((s, 1))
	ucondvar = np.real(invar @ (ucondev ** 2))

	return condvar, ucondvar


def mrkchcor(pmtx, tval):
	"""
	Finds 1st autovariance and serial correlation of Markov Chain
		Inputs:
			pmtx:       kxk transition matrix
						rows denote time t values, cols time t+1 values
			tval:       kx1 vector of values for variable
		Output:
			autocov1:   scalar of 1st autocovariance
			serlcor:    scalar of serial correlation
	"""
	s = pmtx.shape[0]
	# ucondexp = 0
	# ucondvar = -1
	autocov1 = 0
	serlcor = -99

	isgood = chkchain(pmtx)
	if isgood == 0:
		return autocov1, serlcor

	isgood = chktval(pmtx, tval)
	if not isgood:
		return autocov1, serlcor

	invar = mrkinvar(pmtx)
	if np.allclose(invar, -np.ones((1, s))):
		return autocov1, serlcor

	ucondexp = np.dot(invar, tval)

	ucondev = tval - ucondexp
	ucondvar = np.dot(invar, ucondev ** 2)

	autocov1 = np.dot(np.dot((invar * ucondev).T, pmtx), ucondev)
	serlcor = autocov1 / ucondvar

	return autocov1, serlcor


def markch(pmtx, tval):
	"""
	Markch: Finds invariant distribution, conditional and unconditional
        expectations, unconditional variance and unconditional serial
        correlation for a finite state Markov Chain.

        Inputs:
            pmtx:   kxk transition matrix
                    rows denote time t values, cols time t+1 values
            tval:   kx1 vector of values

        Outputs:
            invar:      kxk matrix with each row the invariant distribution
            condexp:    kx1 vector of conditional expectations
            ucondexp:   kx1 vector of unconditional expectations
            condvar:    kx1 vector of conditional variances
            ucondvar:   kxl of unconditional variances
            autocov:    scalar of 1st autocovariance
	"""
	s = pmtx.shape[0]
	invar = np.zeros((1, s))
	condexp = np.ones((s, 1))
	ucondexp = np.ones((s, 1))
	condvar = np.zeros((s, 1))
	condev = np.zeros((s, s))
	ucondvar = np.zeros((1, 1))
	serlcor = 0

	isgood = chkchain(pmtx)
	if isgood == 0:
		return invar, condexp, ucondexp, condvar, ucondvar, serlcor

	isgood = chktval(pmtx, tval)
	if not isgood:
		return invar, condexp, ucondexp, condvar, ucondvar, serlcor

	invar = mrkinvar(pmtx)
	if np.allclose(invar, -np.ones((1, s))):
		return invar, condexp, ucondexp, condvar, ucondvar, serlcor

	condexp = np.dot(pmtx, tval)
	ucondexp = np.dot(invar, tval)

	# Conditional Deviations
	condev = np.outer(tval, np.ones((1, s))) - np.outer(condexp.flatten(), np.ones(s))
	condevsq = condev ** 2

	for j in range(s):
		condvar[j, 0] = np.dot(pmtx[j, :], condevsq[:, j])

	ucondev = tval - ucondexp
	ucondvar = np.dot(invar, (ucondev ** 2))

	autocov1 = np.dot(np.dot((invar * ucondev).T, pmtx), ucondev)
	serlcor = autocov1 / ucondvar

	print("Possible Values: ", tval)
	print("Transition Matrix [i,j] (i->t,j->t+1): ", pmtx)
	print("Invariant Distribution: ", invar)
	print("Conditional Expectations: ", condexp.T)
	print("Conditional Variance: ", condvar.T)
	print("Unconditional Expectations: ", ucondexp)
	print("Unconditional Variance: ", ucondvar)
	print("Unconditional Std. Deviation:", np.sqrt(ucondvar))
	print("1st Autocovariance: ", autocov1)
	print("Serial Correlation: ", serlcor)

	return invar, condexp, ucondexp, condvar, ucondvar, autocov1


if __name__ == "__main__":
	# Example usage:
	pmtx = np.array([[0.2, 0.2, 0.6],
					 [0.5, 0.1, 0.4],
					 [0.3, 0.4, 0.3]])
	tval = np.array([0.10, 0.3, 0.2])

	fdchain(pmtx, tval)
	mrkchvar(pmtx, tval)
	markch(pmtx, tval)
