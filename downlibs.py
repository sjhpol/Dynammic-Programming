import numpy as np
import time


def AMOEBA(p, ftol, maxsec, maxit, fct, prnum):

	date1 = time.time()
	alp, bet, gam = 1.0, 0.5, 2.0
	ndim = p.shape[1]
	npts = ndim + 1
	y = np.zeros((npts, 1))
	for j in range(npts):
		y[j, 0] = fct(p[j, :])

	iter_ = 0
	while True:
		tim = time.time() - date1
		ind = np.argsort(y.flatten())
		ihi = ind[-1]
		inhi = ind[-2]
		ilo = ind[0]

		if abs(y[ihi] + y[ilo]) > 1e-15:
			rtol = 2.0 * abs(y[ihi] - y[ilo]) / abs(y[ihi] + y[ilo])
		else:
			rtol = 2.0 * abs(y[ihi] - y[ilo]) / (1e-15)

		monit(p, y, tim, iter_, prnum)

		if rtol < ftol:
			monit(p, y, tim, iter_, prnum)
			return p, y, iter_, tim

		if iter_ == maxit:
			monit(p, y, tim, iter_, prnum)
			print("Maximum number of iterations exceeded")
			return p, y, iter_, tim

		if tim >= maxsec:
			monit(p, y, tim, iter_, prnum)
			print("Maximum number of seconds exceeded")
			return p, y, iter_, tim

		iter_ += 1
		pbar = np.mean(p, axis=0)
		pr = (1.0 + alp) * pbar - alp * p[ihi, :]
		ypr = fct(pr)

		if ypr <= y[ilo]:
			prr = gam * pr + (1.0 - gam) * pbar
			yprr = fct(prr)
			if yprr < y[ilo]:
				p[ihi, :] = prr
				y[ihi] = yprr
			else:
				p[ihi, :] = pr
				y[ihi] = ypr
		elif ypr >= y[inhi]:
			if ypr < y[ihi]:
				p[ihi, :] = pr
				y[ihi] = ypr
			prr = bet * p[ihi, :] + (1.0 - bet) * pbar
			yprr = fct(prr)
			if yprr < y[ihi]:
				p[ihi, :] = prr
				y[ihi] = yprr
			else:
				p = 0.5 * (p + p[ilo, :])
				for i in range(npts):
					if i != ilo:
						y[i] = fct(p[i, :])
		else:
			p[ihi, :] = pr
			y[ihi] = ypr

	return p, y, iter_, tim


def monit(p, y, tim, iter_, prnum):
	if iter_ % prnum == 0:
		print("iteration:", iter_, "hours elapsed:", tim / 3600)
		np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
		print("Value of function at current simplex:", y.flatten())
		print("current simplex (transposed):", p)
		np.set_printoptions()  # Reset print options
