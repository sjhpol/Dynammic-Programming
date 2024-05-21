import numpy as np
from variables import *
from scipy.stats import norm
from functions import logitrv, logit


def getchrt(data, cohorts_j):
	rn = len(data)
	chrtnum_j = len(cohorts_j) - 1
	chrtcnts = np.zeros((chrtnum_j, 4))
	chrtcnts[:, 0] = np.arange(1, chrtnum_j + 1)
	chrtcnts[:, 1] = cohorts_j[:chrtnum_j] + 1
	chrtcnts[:, 2] = cohorts_j[1:chrtnum_j + 1]

	# Remove Missing Observations
	data[np.where(np.isnan(data))] = mvcode
	data = np.column_stack([data, np.arange(1, rn + 1)])
	srtddata = data[data[:, 0].argsort()]
	k = np.sum(srtddata[:, 0] == mvcode)
	if k > 0:
		srtd2 = srtddata[:k]
	else:
		srtd2 = np.array([])
	srtddata = srtddata[k:]

	rn1 = rn - k
	chrttype = np.ones(rn1)
	cno = 0
	agevec = srtddata[:, 0]

	h = chrtnum_j
	while h >= 1:
		cn = agevec > cohorts_j[h-1]
		cn = np.sum(cn)
		if cn == cno:
			h -= 1
			continue
		rn2 = cn - cno
		rn3 = rn1 - cn
		chrttype[rn3:rn3+rn2] = h
		cno = cn
		h -= 1

	chrttype = chrttype * (1 - (agevec > cohorts_j[chrtnum_j])) * (1 - (agevec <= cohorts_j[0]))

	chrtcnts[:, 3] = [np.sum(chrttype == h) for h in range(1, chrtnum_j + 1)]

	if k > 0:
		chrttype = np.hstack((np.zeros(k), chrttype))
		data = np.vstack((srtd2, srtddata))
	else:
		data = srtddata
	data = np.hstack((data, chrttype.reshape(-1, 1)))
	data = data[np.argsort(data[:, 1])]
	chrttype = data[:, 2]

	return chrtcnts, chrttype


def getmean(data, wgts):
	mns = np.zeros((1, 2))
	cndmnum = np.zeros((1, 2))

	data_ = np.column_stack((data, wgts))  # Combine data and weights
	# data = data[~np.isnan(data)]  # Remove missing observations
	data_ = data_[~np.isnan(data_[:, 0])]
	# data = data[~np.isnan(data).any()]
	if np.isnan(data_).any():  # Check if there are still missing values after removing
		mns[0, 1] = -1e6
	elif np.max(wgts) - np.min(wgts) == 0:  # If all weights are the same
		mns[0, 1] = np.mean(data_[:, 0], axis=0)
		cndmnum[0, 1] = data_.shape[0]
	else:
		wgts = data_[:, 1]
		wgts /= np.sum(wgts)  # Normalize weights to sum to 1
		data_ = data_[:, 0] * wgts  # Weighted data
		mns[0, 1] = np.sum(data_)
		cndmnum[0, 1] = data_.shape[0]

	return mns.reshape(-1,1), cndmnum.reshape(-1,1)


# TODO: REDO WITH getquant
def getcorrs(totcap, ykratio, nkratio, gikratio, debtasst, CAratio, dkratio, divgrowth,
             eqinject, DVFEratio, obsmat, iobsmat, CAobsmat, divobsmat, dvgobsmat, dkobsmat, wgts):
	lbl = ["Total Capital       ", "Output/Capital      ", "Int. Goods/Capital  ",
		   "Gross Invst/Capital ", "Debt/Assets         ", "Cash/Assets         ",
		   "Equity Injections   ", "Dividends/Capital   ", "Dividend Growth     ",
		   "Dividends/DVFE      "]

	cn = len(totcap[0])
	numvars = 10
	krwgts = np.ravel(wgts)
	krwgts2 = np.ravel(wgts[:, 1:cn])

	means = np.zeros(numvars)
	medns = means.copy()
	stddevs = means.copy()
	variances = np.column_stack((means, means))
	autocrrls = means.copy()
	crosscrrls = np.zeros((numvars, numvars))
	alldevs = np.zeros([len(krwgts), numvars])

	for iVar in range(numvars):
		if iVar == 0:
			thisvar = totcap.copy()
			thisobs = obsmat.copy()
		elif iVar == 1:
			thisvar = ykratio.copy()
			thisobs = obsmat.copy()
		elif iVar == 2:
			thisvar = nkratio.copy()
			thisobs = obsmat.copy()
		elif iVar == 3:
			thisvar = gikratio.copy()
			thisobs = iobsmat.copy()
		elif iVar == 4:
			thisvar = debtasst.copy()
			thisobs = obsmat.copy()
		elif iVar == 5:
			thisvar = CAratio.copy()
			thisobs = CAobsmat.copy()
		elif iVar == 6:
			thisvar = eqinject.copy()
			thisobs = obsmat.copy()
		elif iVar == 7:
			thisvar = dkratio.copy()
			thisobs = dkobsmat.copy()
		elif iVar == 8:
			thisvar = divgrowth.copy()
			thisobs = dvgobsmat.copy()
		elif iVar == 9:
			thisvar = DVFEratio.copy()
			thisobs = divobsmat.copy()
		else:
			raise ValueError

		# TODO: TEST ALT NEDENFOR
		# thisvar = thisvar * (1 - np.isnan(thisobs))
		thisvar[thisvar == -99] = np.nan
		# tmn, tcnt = np.mean(np.ravel(thisvar), weights=krwgts, returned=True)
		tmn, tcnt = getmean(np.ravel(thisvar), krwgts)
		means[iVar] = tmn[1]
		# tmn, tcnt = np.quantile(np.ravel(thisvar), 0.5, weights=krwgts, method='interpolated_inverted_cdf')
		tmn, tcnt = getquant(np.ravel(thisvar), krwgts, 0.5, checktie)
		medns[iVar] = tmn[1]

		thesemeans = np.zeros(cn)

		for iMean in range(cn):
			# tmn, tcnt = np.mean(thisvar[:, iMean], weights=wgts[:, iMean], returned=True)
			tmn, tcnt = getmean(thisvar[:, iMean], wgts)
			thesemeans[iMean] = tmn[1]

		thesedevs = thisvar - thesemeans
		alldevs_i = np.ravel(thesedevs)
		np.column_stack([alldevs, alldevs_i])
		alldevs[:, iVar] = alldevs_i
		# tmn, tcnt = np.mean(alldevs_i ** 2, weights=krwgts, returned=True)
		tmn, tcnt = getmean(alldevs_i ** 2, krwgts)
		stddevs[iVar] = np.sqrt(tmn[1])
		variances[iVar, 0] = tmn[1]

		thisobs = np.ravel(thisobs[:, 1:cn]) * np.ravel(thisobs[:, :cn-1])
		crrntdevs = np.ravel(thesedevs[:, 1:cn])   # * (1 - np.isnan(thisobs))
		crrntdevs[crrntdevs == -99] = np.nan

		laggddevs = np.ravel(thesedevs[:, :cn-1])   # * (1 - np.isnan(thisobs))
		thesedevs[thesedevs == -99] = np.nan
		# tmn, tcnt = np.mean(crrntdevs ** 2, weights=krwgts2, returned=True)
		tmn, tcnt = getmean(crrntdevs ** 2, krwgts2)
		var2 = np.sqrt(tmn[1])
		# tmn, tcnt = np.mean(laggddevs ** 2, weights=krwgts2, returned=True)
		tmn, tcnt = getmean(laggddevs ** 2, krwgts2)
		var2 *= np.sqrt(tmn[1])
		variances[iVar, 1] = var2

		# tmn, tcnt = np.mean(crrntdevs * laggddevs, weights=krwgts2, returned=True)
		tmn, tcnt = getmean(crrntdevs * laggddevs, krwgts2)
		autocrrls[iVar] = tmn[1] / variances[iVar, 1]

		for jVar in range(numvars):
			thisvar = alldevs[:, iVar]
			thatvar = alldevs[:, jVar]
			# tmn, tcnt = np.mean(thisvar * thatvar, weights=krwgts, returned=True)
			tmn, tcnt = getmean(thisvar * thatvar, krwgts)
			crosscrrls[iVar, jVar] = tmn[1] / (stddevs[iVar] * stddevs[jVar])

	print("                         Means      Medians   Std. Devs   Autocrrls   |---Crosscrrls --->")
	for iVar in range(numvars):
		print(lbl[iVar], means[iVar], medns[iVar], stddevs[iVar], autocrrls[iVar], crosscrrls[iVar, :])

	return


def getquant(data, wgts, quants, checktie):
	if isinstance(quants, float) or isinstance(quants, int):
		quants = np.array([quants])
	qnum = len(quants)
	rn = data.shape[0]
	qnts = np.zeros((qnum, 2))
	qnts[:, 0] = quants
	cndmnum = np.zeros((qnum + 1, 2))
	cndmnum[:, 0] = np.concatenate((quants, [1]))  # Concatenate quants with 1 for qnum+1
	# Remove missing observations
	srtddata = np.column_stack((data, wgts))  # Combine data with weights
	srtddata = srtddata[~np.isnan(srtddata[:, 0])]  # Remove missing observations
	srtddata = srtddata[srtddata[:, 0].argsort()]  # Sort data based on first column
	rn1 = srtddata.shape[0]  # Number of non-missing observations
	cdf = np.cumsum(srtddata[:, 1], axis=0) / np.sum(srtddata[:, 1], axis=0)  # Cumulative distribution based on weights
	qno = 0

	for i in range(qnum):
		qn = np.sum(cdf < quants[i]) + 1
		qnt = srtddata[qn, 0]
		qnts[i, 1] = qnt

		if checktie == 1:
			j = np.sum(srtddata[qn:rn1, 0] == qnt) - 1
			qn += j

		if qn == qno:
			continue

		rn2 = qn - qno
		cndmnum[i, 1] = rn2
		qno = qn

	if qn < rn1:
		cndmnum[qnum, 1] = rn1 - qn

	return qnts.reshape(-1, 1), cndmnum.reshape(-1, 1)


def getqunt2(data_, quants_):
	if isinstance(quants_, float) or isinstance(quants_, int):
		quants_ = np.array([quants_])
	qnum = len(quants_)
	qnts = np.zeros((qnum, 2))
	qnts[:, 0] = quants_
	cndmnum = np.zeros((qnum + 1, 2))
	cndmnum[:, 0] = np.concatenate((quants_, [1]))

	data_ = data_[~np.isnan(data_)]
	rn = len(data_)

	if rn > 1:
		qnts[:, 1] = np.quantile(data_, quants_, method='interpolated_inverted_cdf')
	else:
		qnts[:, 1] = -1e6

	qno = 0
	for i in range(qnum):
		qn = np.sum(data_ <= qnts[i, 1])
		if qn == qno:
			continue

		rn2 = qn - qno  # Counts for interval (q_i-1,q_i]
		cndmnum[i, 1] = rn2
		qno = qn

	# Count observations above highest quantile
	if qn < rn:
		cndmnum[qnum, 1] = rn - qn

	return qnts, cndmnum


def getqunts(chrttype_j, FStype_j, datamat_j, obsmat_j, quants_j, mmtyrs_j, wgts):
	FSnum_j = np.max(FStype_j, axis=0)
	chrtnum_j = np.max(chrttype_j, axis=0)
	if isinstance(quants_j, float) or isinstance(quants_j, int):
		qnum_j = 1
	else:
		qnum_j = len(quants_j)

	chrtnum_j, FSnum_j, qnum_j, mmtyrs_j = int(chrtnum_j), int(FSnum_j), int(qnum_j), int(mmtyrs_j)
	recsize = (chrtnum_j, FSnum_j, qnum_j, mmtyrs_j)
	dataprfs = np.full(recsize, mvcode).astype(float)
	datacnts = np.zeros(recsize)

	for iChrt in range(1, chrtnum_j + 1):
		for iFS in range(1, FSnum_j + 1):
			indicat0 = (chrttype_j == iChrt) * (FStype_j == iFS)
			indicat = obsmat_j * indicat0.reshape(-1, 1)
			print("Cohort =", iChrt, "FS Quintile =", iFS, np.mean(indicat, axis=0))

			#data = datamat_j * (1 - np.isnan(indicat))
			indicat = indicat.astype(float)
			indicat[indicat == 0] = np.nan
			data = datamat_j * indicat
			# data[data == 0] = np.nan
			# datamat_j[datamat_j == -99] = np.nan
			# data = datamat_j

			for iYear in range(1, mmtyrs_j + 1):
				if np.sum(indicat[:, iYear - 1] > 0) < quantmin:
					continue

				if quants_j == 0:
					tempprf, tempnum = getmean(data[:, iYear - 1], wgts[:, iYear - 1])
				else:
					if (np.max(wgts[:, iYear - 1]) - np.min(wgts[:, iYear - 1])) == 0:
						tempprf, tempnum = getqunt2(data[:, iYear - 1], quants_j)
					else:
						tempprf, tempnum = getquant(data[:, iYear - 1], wgts[:, iYear - 1], quants_j, checktie)

				for iQunt in range(1, qnum_j + 1):
					if quants_j == 0:
						dataprfs[iChrt - 1, iFS - 1, iQunt - 1, iYear - 1] = tempprf[1, 0]
						datacnts[iChrt - 1, iFS - 1, iQunt - 1, iYear - 1] = np.sum(tempnum[:, 0])
					else:
						dataprfs[iChrt - 1, iFS - 1, iQunt - 1, iYear - 1] = tempprf[iQunt - 1, 1]
						datacnts[iChrt - 1, iFS - 1, iQunt - 1, iYear - 1] = np.sum(tempnum[:, 1])

	return dataprfs, datacnts


def setindx1(numpts, gridmin, gridmax, grdcoef):
	indnum = np.arange(2, numpts+1)
	gridrat = (indnum - 1) / (numpts - 1)
	indx = gridmin + np.power(gridrat, 1 / grdcoef) * (gridmax - gridmin)
	indx = np.concatenate(([gridmin], indx))

	return indx


def getgrids(Arange, Afine, grdcoef_lv, grdcoef_rt, Enum0, TAnum0,
			 nodebt, Knum, lagKnum, Bnum0, NKnum0, Cnum0, K_min, prnres):

	# A indicates financial assets
	A_min = 0
	A_max = Arange + A_min  	# maximum asset level, in (000s)
	Anum = int(Afine * 500) + 1
	Astate = setindx1(Anum,A_min,A_max,grdcoef_lv)
	Aindvec = np.arange(1, Anum + 1)

	# E indicates net worth
	E_min = 0
	E_max = A_max  	# maximum equity, in (000s)
	Estate = setindx1(Enum0,E_min,E_max,0.65*grdcoef_lv)
	Estate = np.concatenate((-np.flip(Estate[1:int(0.4 * Enum0)]), Estate))
	E_min = np.min(Estate, axis=0)
	Enum = len(Estate)
	Eindvec = np.arange(1, Enum + 1)

	# TA indicates total assets
	TA_min = A_min
	TA_max = A_max
	TAnum = TAnum0
	TAstate = setindx1(TAnum0,TA_min,TA_max,grdcoef_lv)
	TAstate = np.concatenate((-np.flip(TAstate[1:int(TAnum0 / 4)]), TAstate))
	TA_min = np.min(TAstate, axis=0)
	TAnum = TAstate.shape[0]
	TAindvec = np.arange(1, TAnum + 1)

	# B indicates debt
	B_min = 0
	B_max = TA_max
	Bnum = Bnum0
	Bstate = setindx1(Bnum,B_min,B_max,grdcoef_lv)
	if nodebt == 1:
		B_min = np.min(Bstate, axis=0)
		B_max = np.min(Bstate, axis=0)
		Bstate = np.array([B_min, B_max])

	Bnum = Bstate.shape[0]
	Bindvec = np.arange(1, Bnum + 1)

	# K indicates physical capital
	K_max = A_max
	Kstate = setindx1(Knum,K_min,K_max,grdcoef_lv)
	Kindvec = np.arange(1, Knum + 1)
	lagKstate = setindx1(lagKnum,K_min,K_max,grdcoef_lv)  	# Used to find capital adj costs

	# NK indicates intermediate goods/capital
	NK_min = 0
	NK_max = 1.5
	NKstate = setindx1(NKnum0,NK_min,NK_max,grdcoef_rt)
	NKstate = NKstate[1:NKnum0]
	NKnum = NKstate.shape[0]

	# NKstate is relative to static optimum.  Make grid finer around value of 1
	dN = 2
	iNKmin = int(0.4 * len(NKstate))
	iNKmax = int(0.8 * len(NKstate))
	NKstate2 = NKstate[:iNKmin]
	iNK = iNKmin
	while iNK < iNKmax:
		NKfill = np.linspace(NKstate[iNK-1], NKstate[iNK], dN + 1)[1:]
		NKstate2 = np.concatenate((NKstate2, NKfill))
		iNK += 1

	NKstate = np.concatenate((NKstate2, NKstate[iNKmax:NKnum]))
	iNK = np.sum(NKstate < 1, axis=0)
	NKfill = np.linspace(NKstate[iNK], NKstate[iNK + 1], dN + 1)[1:-1]
	NKstate = np.concatenate((NKstate[:iNK], NKfill, NKstate[iNK + 1:]))
	NKnum = len(NKstate)
	NKindvec = np.arange(1, NKnum + 1)

	C_min = A_min
	C_max = A_max * 0.19
	Cnum = int(Cnum0 * 2 / 3)
	Cstate = setindx1(Cnum,C_min,C_max,grdcoef_lv*.8)
	C_min = C_max
	C_max = 0.8 * A_max
	Cnum = Cnum0 - Cnum + 1
	Cfill = setindx1(Cnum,C_min,C_max,grdcoef_lv)[1:]
	Cstate = np.concatenate((Cstate, Cfill))
	c_min = Cstate[0]
	Cnum = len(Cstate)
	Cindvec = np.arange(1, Cnum + 1)

	if prnres > 1:
		print("Financial Asset Grid", np.column_stack((Aindvec, Astate)))

	if prnres > 0:
		print("Net Worth Grid", np.column_stack((Eindvec, Estate)))
		print("Total Asset Grid", np.column_stack((TAindvec, TAstate)))
		print("Debt Grid", np.column_stack((np.arange(1, Bnum + 1), Bstate)))
		print("Capital Choice Grid", np.column_stack((Kindvec, Kstate)))
		print("iGoods relative ratio Grid", np.column_stack((NKindvec, NKstate)))
		print("Cash Choice Grid", np.column_stack((Cindvec, Cstate)))

	sizevec = [A_min, A_max, E_min, E_max, TA_min, TA_max, B_min, B_max, K_min, K_max, C_min, C_max]
	numvec = [Anum, Enum, TAnum, Bnum, Knum, lagKnum, Cnum]

	return sizevec, numvec, Astate, Estate, TAstate, Bstate, Kstate, lagKstate, NKstate, Cstate


def makepvecs(parmvec, betamax, linprefs, nobeq, w_0, bigR, numFTypes, inadaU, nonshft, noDScost, nofcost,
		  	nocolcnst, prnres, noReneg, finparms):
	bta = betamax * logit(parmvec[0])
	nu = np.exp(parmvec[1]) * (1 - linprefs)
	if nu == 1:
		nu = 0.9999
	nu2 = 1 - nu
	c_0 = np.exp(parmvec[2]) * (1 - inadaU)
	c_bar = np.exp(parmvec[3]) 	# consumption threshold for bequest motive
	# strength of the bequest function. 1 => no bequest motive; 0 => an infinite one
	finalMPC = logit(parmvec[4])
	if nobeq == 1:
		finalMPC = 1
	chi0 = logit(parmvec[5])
	cfloor = np.exp(parmvec[6])
	_F = (1 - finalMPC) / finalMPC
	if nu == 0:
		_L = 0
	else:
		_L = (bigR * bta) ** (-1 / nu)
	thetaB = _F * _L * bigR
	# Flow utility from being a farmer, as consumption increment
	chi = chi0 * (w_0 + c_0)
	# Avoid divide by zero errors below
	if _F == 0 or _L == 0:
		c_1 = c_bar
	else:
		c_1 = (c_bar + c_0) / _L
	# Coefficient on owner labor
	alp = logit(parmvec[7:9])
	gam0 = logit(parmvec[9:11])
	if numFTypes == 1:
		alp[1] = alp[0]
		gam0[1] = gam0[0]
	# Coefficient on capital (including land)
	gam = (1 - alp) * gam0
	# coefficient on intermediate goods
	ag2 = np.ones(2) - alp - gam
	nshft = np.exp(parmvec[11]) * (1 - nonshft)
	# liquidation costs
	lam = logit(parmvec[12])
	phi = logit(parmvec[13]) * (1 - noDScost)
	# liquidity constraint scaler
	zta = np.exp(parmvec[14])
	# fixed operating cost
	fcost = np.exp(parmvec[15]) * (1 - nofcost)
	# colcnst*B <= K
	colcnst = np.exp(parmvec[16]) * (1 - 2 * nocolcnst)

	prefparms = np.concatenate([np.array([x]) for x in [bta, nu, nu2, c_0, c_1, thetaB, chi, cfloor]])
	finparms = np.concatenate((alp, gam, np.array([nshft]), np.array([lam]), np.array([phi]), np.array([zta]), np.array([dlt]), np.array([fcost]), np.array([colcnst]), finparms))

	if prnres > 0:
		prnparms(nu, c_0, bta, cfloor, finalMPC, bigR, thetaB, c_1, chi, alp, gam, nshft, lam, phi, zta, dlt, fcost,
				 colcnst, noReneg, w_0)

	return prefparms, finparms, gam, ag2, nshft, fcost


def prnparms(nu, c_0, bta, cfloor, finalMPC, bigR, thetaB, c_1, chi, alp, gam, nshft, lam, phi, zta, dlt, fcost,
			 colcnst, noReneg, w_0):
	print("Coefficient of relative risk aversion =", nu)
	print("Flow Utility shifter =", c_0)
	print("Discount factor =", bta)
	print("Gross rate of return =", bigR)
	print("Consumption floor (in $000s) =", cfloor)
	print("MPC out of terminal wealth =", finalMPC)
	print("Bequest parameter thetaB =", thetaB)
	print("Bequests become operative at (in $000s) =", c_1)
	print("Coefficient on own labor (alpha) =", alp)
	print("Coefficient on capital (gamma) =", gam)
	print("Utility from farming (consump. flow) =", chi)
	print("Shift term for igoods =", nshft)
	print("Percentage loss for liquidation =", lam)
	print("Percentage loss for capital downsizing =", phi)
	print("Cash leverage ratio (zeta) =", zta)
	print("Net depreciation rate =", dlt)
	print("Fixed operating cost =", fcost)
	print("1 / Fraction of K collateralizable =", colcnst)
	print("Defaulters can renegotiate =", 1 - noReneg)
	print("Outside wage =", w_0)


def tauch(stdi, rho, M, utail, ltail):
	"""
	Tauch:    Converts an AR(1) process into a Markov chain
			  Follows Tauchen (Ec. Letters, 1986)
	Inputs:
		stdi:       Scalar, std. dev. of innovation
		rho:        Scalar, AR(1) correlation coefficient
		M:          Scalar indicating number of elements in the chain
		Tail:       Scalar indicating percentages left in each tail
	Output:
		pmtx:       MxM transition matrix
					rows denote time t values, cols time t+1 values
		values:     Mx1 vector of values
		intvals:    (M-1)x1 vectors of halfway points on values
	"""
	stdy = np.sqrt(stdi ** 2 / (1 - rho ** 2))
	v_min = norm.ppf(ltail) * stdy
	v_max = norm.ppf(1 - utail) * stdy

	pmtx = np.zeros((M, M))
	values = np.linspace(v_min, v_max, M)
	intvals = (values[:-1] + values[1:]) / 2

	for i in range(M):
		pyold = rho * values[i]
		pmtx[i, 0] = norm.cdf((intvals[0] - pyold) / stdi)  # lower tail

		for j in range(1, M - 1):
			pmtx[i, j] = norm.cdf((intvals[j] - pyold) / stdi) - norm.cdf((intvals[j - 1] - pyold) / stdi)

		pmtx[i, M - 1] = 1 - norm.cdf((intvals[M - 2] - pyold) / stdi)  # upper tail

	return pmtx, values, intvals


def FSquant(data, wgts, quants, checktie):
	if isinstance(quants, int) or isinstance(quants, float):
		quants = np.array([quants])
	qnum = len(quants)
	rn = data.shape[0]

	if quants[qnum - 1] == 0:
		qnts = np.array([1, np.max(data)])
		cndmnum = np.array([1, rn])
		qntype = np.ones(rn)
		return qnts, cndmnum, qntype

	qnts = np.zeros((qnum, 2))
	qnts[:, 0] = quants
	cndmnum = np.zeros((qnum + 1, 2))
	cndmnum[:, 0] = np.concatenate((quants, [1]))

	# Remove Missing Observations
	data = data[data[:, 0] != mvcode]
	data = np.column_stack([data, wgts, np.arange(1,rn+1)])
	srtddata = data[data[:, 0].argsort()]

	k = np.sum(srtddata[:, 0] == mvcode)
	if k > 0:
		srtd2 = srtddata[:k, :]
	elif k == 0:
		srtd2 = np.array([])

	srtddata = srtddata[k:rn, :]

	rn1 = rn - k
	cdf = srtddata[:, 1]
	cdf = np.cumsum(cdf) / np.sum(cdf)

	qno = 0
	qntype = np.ones(rn1)

	for i in range(qnum):
		qn = cdf < quants[i]
		qn = np.sum(qn) + 1
		qnt = srtddata[qn - 1, 0]
		qnts[i, 1] = qnt

		if checktie == 1:
			j = np.sum(srtddata[qn-1:, 0] == qnt) - 1
			qn += j

		if qn == qno:
			continue

		rn2 = qn - qno
		cndmnum[i, 1] = rn2
		qntype[qno:qn] = (i + 1) * np.ones(rn2)
		qno = qn

	if qn < rn1:
		qntype[qn:] = (qnum + 1) * np.ones(rn1 - qn)
		cndmnum[qnum, 1] = rn1 - qn

	if k > 0:
		qntype = np.zeros(k).tolist() + qntype.tolist()
		data = np.vstack([srtd2, srtddata])
	else:
		data = srtddata

	data = np.column_stack([data, qntype])
	data = data[data[:, 2].argsort()]
	qntype = data[:, -1]

	return qnts, cndmnum, qntype


# if __name__ == "__main__":
# 	mvcode = -99
# 	checktie = 1
# 	quantmin = 10
#
# 	# Example usage:
# 	Arange = 10
# 	Afine = 0.1
# 	grdcoef_lv = 0.5
# 	grdcoef_rt = 0.5
# 	Enum0 = 100
# 	TAnum0 = 100
# 	nodebt = 0
# 	Knum = 50
# 	lagKnum = 50
# 	Bnum0 = 100
# 	NKnum0 = 100
# 	Cnum0 = 100
# 	K_min = 0
# 	prnres = 1
# 	getgrids(Arange, Afine, grdcoef_lv, grdcoef_rt, Enum0, TAnum0, nodebt, Knum, lagKNum, Bnum0, NKnum0, Cnum0, K_min, prnres)

