from markch import markch
from variables import *
from functions import logitrv, logit

import numpy as np
import os

import pandas as pd
from scipy.stats import norm


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
			indicat0 = (chrttype_j == iChrt).reshape(-1,1) * (FStype_j == iFS).reshape(-1,1)
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
						tempprf = tempprf.reshape((len(np.array([quants_j])), 2))
						tempnum = tempnum.reshape((len(np.array([quants_j])) + 1, 2))

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
		  	nocolcnst, prnres, noReneg, finparms0):
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
	finparms = np.concatenate((alp, gam, np.array([nshft]), np.array([lam]), np.array([phi]), np.array([zta]), np.array([dlt]), np.array([fcost]), np.array([colcnst]), finparms0))

	if prnres > 0:
		prnparms(nu, c_0, bta, cfloor, finalMPC, bigR, thetaB, c_1, chi, alp, gam, nshft, lam, phi, zta, dlt, fcost,
				 colcnst, noReneg, w_0)

	return prefparms, finparms, gam, ag2, nshft, fcost


def makepvecs2(parmvec, linprefs, w_0, bigR, inadaU, nonshft, noDScost, nofcost,
		  	nocolcnst, prnres, noReneg, finparms0):

	bta = parmvec[0]
	nu = parmvec[1] * (1 - linprefs)
	if nu == 1:
		nu = 0.9999
	nu2 = 1 - nu
	c_0 = parmvec[2] * (1 - inadaU)
	c_1 = parmvec[3]
	thetaB = parmvec[4]
	chi = parmvec[5]
	cfloor = parmvec[6]
	alp = parmvec[7:9]
	gam = parmvec[9:11]
	ag2 = np.ones(2) - alp - gam
	nshft = parmvec[11] * (1 - nonshft)
	lam = parmvec[12]
	phi = parmvec[13] * (1 - noDScost)
	zta = parmvec[14]
	fcost = parmvec[15] * (1 - nofcost)
	colcnst = parmvec[16] * (1 - 2 * nocolcnst)

	prefparms = np.array([bta, nu, nu2, c_0, c_1, thetaB, chi, cfloor])
	finparms = np.concatenate((alp, gam, [nshft, lam, phi, zta, dlt, fcost, colcnst], finparms0))

	if thetaB > 0:
		_L = (bta * bigR) ** (-1 / nu)
		MPC = _L * bigR / (thetaB + _L * bigR)
	else:
		MPC = 1

	if prnres > 0:
		prnparms(nu, c_0, bta, cfloor, MPC, bigR, thetaB, c_1, chi, alp, gam, nshft, lam, phi, zta, dlt, fcost, colcnst,
				 noReneg, w_0)

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


def missingvec(tkqnt, divqnt, dvgqnt, DAqnt, CAqnt, nkqnt, gikqnt, ykqnt,
			   obsavg, tkqcnts, divqcnts, dvgqcnts):

	tkprof = arraytovec(IDmissing(tkqnt, tkqcnts, 0, 0))

	if divmmts == 1:
		divprof = arraytovec(IDmissing(divqnt, divqcnts, 0, 0))
	elif divmmts == 2:
		divprof = arraytovec(IDmissing(dvgqnt, dvgqcnts, 1, 0))
	else:
		divprof = np.array([])

	DAprof = arraytovec(IDmissing(DAqnt, tkqcnts, 1, 0))
	CAprof = arraytovec(IDmissing(CAqnt, tkqcnts, 1, 0))
	nkprof = arraytovec(IDmissing(nkqnt, tkqcnts, 0, 0))
	gikprof = arraytovec(IDmissing(gikqnt, tkqcnts, 0, 0))
	ykprof = arraytovec(IDmissing(ykqnt, tkqcnts, 0, 0))
	obsprof = arraytovec(IDmissing(obsavg, tkqcnts, 0, 0))

	missvec = np.concatenate((tkprof, divprof, DAprof, CAprof,nkprof, gikprof, ykprof, obsprof))

	obsmmtind = np.ones((8, 2), dtype=int)
	obsmmtind[0, 1] = obsmmtind[0, 0] + np.sum(tkprof == 0) - 1
	obsmmtind[1, 0] = obsmmtind[0, 1] + (divmmts > 0)
	obsmmtind[1, 1] = obsmmtind[1, 0] + np.sum(divprof == 0) - (divmmts > 0)
	obsmmtind[2, 0] = obsmmtind[1, 1] + 1
	obsmmtind[2, 1] = obsmmtind[2, 0] + np.sum(DAprof == 0) - 1
	obsmmtind[3, 0] = obsmmtind[2, 1] + 1
	obsmmtind[3, 1] = obsmmtind[3, 0] + np.sum(CAprof == 0) - 1
	obsmmtind[4, 0] = obsmmtind[3, 1] + 1
	obsmmtind[4, 1] = obsmmtind[4, 0] + np.sum(nkprof == 0) - 1
	obsmmtind[5, 0] = obsmmtind[4, 1] + 1
	obsmmtind[5, 1] = obsmmtind[5, 0] + np.sum(gikprof == 0) - 1
	obsmmtind[6, 0] = obsmmtind[5, 1] + 1
	obsmmtind[6, 1] = obsmmtind[6, 0] + np.sum(ykprof == 0) - 1
	obsmmtind[7, 0] = obsmmtind[6, 1] + 1
	obsmmtind[7, 1] = obsmmtind[7, 0] + np.sum(obsprof == 0) - 1

	return missvec, obsmmtind


def getorders(arr):
	return arr.shape


def arraytovec(thisarray):
	return thisarray.flatten()


def IDmissing(dataprfs, datacnts, skipfirst, skiplast, cellmin=1):
	dimvec = getorders(dataprfs)
	FSnum_j = dimvec[1]
	chrtnum_j = dimvec[0]
	qnum_j = dimvec[2]
	nyears_j = dimvec[3]

	missInd = (dataprfs == mvcode) + (datacnts < cellmin)
	missInd = (missInd > 0)

	for iChrt in range(chrtnum_j):
		for iFS in range(FSnum_j):
			for iQunt in range(qnum_j):
				if skipfirst == 1:
					missInd[iChrt, iFS, iQunt, 0] = True
				if skiplast == 1:
					missInd[iChrt, iFS, iQunt, nyears_j - 1] = True

	return missInd


def makemomvec(tkqnt, divqnt, dvgqnt, DAqnt, CAqnt, nkqnt, gikqnt, ykqnt, obsavg):
	tkprof = arraytovec(tkqnt)

	if divmmts == 1:
		divprof = arraytovec(divqnt)
	elif divmmts == 2:
		divprof = arraytovec(dvgqnt)
	else:
		divprof = np.array([])  # Equivalent to an empty vector

	DAprof = arraytovec(DAqnt)
	CAprof = arraytovec(CAqnt)
	nkprof = arraytovec(nkqnt)
	gikprof = arraytovec(gikqnt)
	ykprof = arraytovec(ykqnt)
	obsprof = arraytovec(obsavg)

	momvec = np.concatenate((tkprof, divprof, DAprof, CAprof, nkprof, gikprof, ykprof, obsprof))

	return momvec


def loadSims(fcost, gam, ag2, nshft, k_0, optNK, timespan, numsims, cashlag, tfplag, divlag, divbasemin, bigG,
			 gkE, dlt, rdgE, mvcode, feshks, prnres):
	def reshape(array, rows, cols):
		return array.reshape(rows, cols).T

	# Load data
	ageS = np.loadtxt('iofiles/ageS.txt', ndmin=2)
	assetsS = np.loadtxt('iofiles/assetsS.txt', ndmin=2)
	cashS = np.loadtxt('iofiles/cashS.txt', ndmin=2)
	debtS = np.loadtxt('iofiles/debtS.txt', ndmin=2)
	divsS = np.loadtxt('iofiles/divsS.txt', ndmin=2)
	equityS = np.loadtxt('iofiles/equityS.txt', ndmin=2)
	expenseS = np.loadtxt('iofiles/expenseS.txt', ndmin=2)
	fracRPS = np.loadtxt('iofiles/fracRPS.txt', ndmin=2)
	intRateS = np.loadtxt('iofiles/intRateS.txt', ndmin=2)
	liqDecS = np.loadtxt('iofiles/liqDecS.txt', ndmin=2)
	outputS = np.loadtxt('iofiles/outputS.txt', ndmin=2)
	totKS = np.loadtxt('iofiles/totKS.txt', ndmin=2)
	ZvalsS = np.loadtxt('iofiles/ZvalsS.txt', ndmin=2)
	obsSim = np.loadtxt('iofiles/obsSim.txt', ndmin=2)
	iobsSim = np.loadtxt('iofiles/iobsSim.txt', ndmin=2)
	dvgobsSim = np.loadtxt('iofiles/dvgobsSim.txt', ndmin=2)
	FEshks = np.loadtxt('iofiles/FEshks.txt', ndmin=2)
	IDsim = np.loadtxt('iofiles/IDsim.txt', ndmin=2)
	ftype_sim = np.loadtxt('iofiles/ftype_sim.txt', ndmin=2)
	simwgts = np.loadtxt('iofiles/simwgts.txt', ndmin=2)

	ageS = ageS.reshape(timespan + 1, numsims).transpose()
	assetsS = assetsS.reshape(timespan + 1, numsims).transpose()
	cashS = cashS.reshape(timespan + 1, numsims).transpose()
	debtS = debtS.reshape(timespan + 1, numsims).transpose()
	divsS = divsS.reshape(timespan + 1, numsims).transpose()
	equityS = equityS.reshape(timespan + 1, numsims).transpose()
	expenseS = expenseS.reshape(timespan + 1, numsims).transpose()
	fracRPS = fracRPS.reshape(timespan + 1, numsims).transpose()
	intRateS = intRateS.reshape(timespan + 1, numsims).transpose()
	liqDecS = liqDecS.reshape(timespan + 1, numsims).transpose()
	outputS = outputS.reshape(timespan + 1, numsims).transpose()
	totKS = totKS.reshape(timespan + 1, numsims).transpose()
	ZvalsS = ZvalsS.reshape(timespan + 1, numsims).transpose()

	ageSim = ageS[:, :timespan]
	assetSim = assetsS[:, :timespan] + 1e-10
	cashSim = cashS[:, cashlag:timespan + cashlag]
	debtSim = debtS[:, :timespan]
	equitySim = equityS[:, :timespan]
	expenseSim = expenseS[:, :timespan]
	fracRPSim = fracRPS[:, :timespan]
	intRateSim = intRateS[:, :timespan]
	outputSim = outputS[:, :timespan]
	ZvalSim = ZvalsS[:, tfplag:timespan + tfplag]
	hetag2 = ag2[ftype_sim.reshape(-1, ).astype(int) - 1]
	hetgam = gam[ftype_sim.reshape(-1, ).astype(int) - 1]
	alp = 1 - hetag2 - hetgam
	optKSim = (k_0.reshape(-1,1) * np.exp(FEshks)) ** (1 / alp.reshape(-1,1))
	divSim = divsS[:, divlag:timespan + divlag]
	fixeddiv = (np.abs(divSim[:, :timespan - 1]) > divbasemin)
	divsign = 1 - 2 * (divSim[:, :timespan - 1] < 0)
	fixeddiv = divSim[:, :timespan - 1] * fixeddiv + divbasemin * (1 - fixeddiv) * divsign
	dvgSim = divSim[:, 1:timespan] / fixeddiv
	dvgSim = np.column_stack((np.zeros((numsims, 1)), dvgSim))
	eqinjSim = divSim
	goteqiSim = eqinjSim > 0

	aliveSim = (liqDecS == 0)
	fwdalivesim = aliveSim[:, 1:timespan + 1]
	NInvSim = (totKS[:, 1:timespan + 1] * fwdalivesim) * bigG / (1 + gkE) - (
				totKS[:, :timespan] * aliveSim[:, :timespan])
	deprSim = totKS[:, :timespan] * aliveSim[:, :timespan] * dlt
	GInvSim = NInvSim + deprSim
	totKSim = totKS[:, :timespan] + 1e-10
	profitSim = outputSim - totKSim * rdgE - expenseSim
	netWorthSim = assetSim - debtSim
	iZvalSim = outputSim / ((totKSim ** hetgam.reshape(-1,1)) * ((expenseSim - fcost + nshft) ** hetag2.reshape(-1,1)))

	liqDecSim = liqDecS[:, :timespan]
	ialiveSim = aliveSim[:, :timespan] * fwdalivesim
	dvgaliveSim = aliveSim[:, divlag:timespan + divlag] * np.column_stack((np.zeros((numsims, 1)), aliveSim[:, divlag:timespan + divlag - 1]))
	aliveSim = aliveSim[:, :timespan]
	exiterrs = (aliveSim == 0).reshape(-1,1) * obsSim
	exiterrs = exiterrs.reshape(numsims, timespan)
	gotxerrs = (exiterrs @ np.ones((timespan, 1))) > 0
	IDlist = np.unique(IDsim)
	iDDums = IDsim == IDlist
	# xerravg = np.linalg.inv(iDDums.T @ iDDums) @ (iDDums.T @ (np.column_stack((gotxerrs, exiterrs))))
	iDDums = None

	ialiveSim = iobsSim.reshape(numsims, timespan) * ialiveSim
	dvgaliveSim = dvgobsSim.reshape(numsims, timespan) * dvgaliveSim
	aliveSim = obsSim * aliveSim.reshape(-1,1)
	aliveSim = aliveSim.reshape(numsims, timespan)
	fwdalivesim = obsSim * fwdalivesim.reshape(-1,1)
	fwdalivesim = fwdalivesim.reshape(numsims, timespan)
	divaliveSim = divlag * fwdalivesim + (1 - divlag) * aliveSim
	DVKalivesim = divaliveSim * aliveSim
	cshaliveSim = cashlag * fwdalivesim + (1 - cashlag) * aliveSim
	CAaliveSim = cshaliveSim * aliveSim

	DAratioSim = (debtSim / assetSim + 1) * aliveSim - 1
	CAratioSim = (cashSim / assetSim + 1) * CAaliveSim - 1
	NKratioSim = (expenseSim / totKSim + 1) * aliveSim - 1
	YKratioSim = (outputSim / totKSim + 1) * aliveSim - 1
	NIKratioSim = (NInvSim / totKSim + 1) * ialiveSim - 1
	GIKratioSim = (GInvSim / totKSim + 1) * ialiveSim - 1
	DVKratioSim = (divSim / totKSim + 1) * DVKalivesim - 1

	DVFEratioSim, DVFESim = removeFE(np.abs(divSim), divaliveSim)
	DVFEratioSim = (divSim / DVFESim.reshape(-1,1) + 1) * aliveSim - 1

	alivesim2 = aliveSim * simwgts.reshape(numsims, timespan)
	ialivesim2 = ialiveSim * simwgts.reshape(numsims, timespan)
	CAalivesim2 = CAaliveSim * simwgts.reshape(numsims, timespan)
	cshalivesim2 = cshaliveSim * simwgts.reshape(numsims, timespan)
	divalivesim2 = divaliveSim * simwgts.reshape(numsims, timespan)

	aliveavg = np.mean(alivesim2, axis=0)
	ialiveavg = np.mean(ialivesim2, axis=0)
	CAaliveavg = np.mean(CAalivesim2, axis=0)
	cshaliveavg = np.mean(cshalivesim2, axis=0)
	divaliveavg = np.mean(divalivesim2, axis=0)

	simavg = np.array([
		aliveavg,
		np.mean(ZvalSim * alivesim2, axis=0) / aliveavg,
		np.mean(assetSim * alivesim2, axis=0) / aliveavg,
		np.mean(DAratioSim * alivesim2, axis=0) / aliveavg,
		np.mean(CAratioSim * CAalivesim2, axis=0) / CAaliveavg,
		np.mean(totKSim * alivesim2, axis=0) / aliveavg,
		np.mean(NKratioSim * alivesim2, axis=0) / aliveavg,
		np.mean(YKratioSim * alivesim2, axis=0) / aliveavg,
		np.mean(NIKratioSim * ialivesim2, axis=0) / ialiveavg,
		np.mean(GIKratioSim * ialivesim2, axis=0) / ialiveavg,
		np.mean(exiterrs * simwgts.reshape(numsims, timespan), axis=0) / aliveavg
	])

	print("      Year    frac alive   TFP shks     Assets   Debt/Assets  Cash/Assets    Capital    igoods/K")
	print("       Y/K    Net Inv/K   Gross I/K   exiterrs")

	for t in range(timespan):
		if t % prnres == 0:
			print(
				f'{t} {simavg[0, t]:.4f} {simavg[1, t]:.4f} {simavg[2, t]:.4f} {simavg[3, t]:.4f} {simavg[4, t]:.4f} {simavg[5, t]:.4f}')
			print(f' {simavg[6, t]:.4f} {simavg[7, t]:.4f} {simavg[8, t]:.4f} {simavg[9, t]:.4f} {simavg[10, t]:.4f}')

	# You might want to save results instead of just printing them, depending on your needs

	# Calculate avxerr
	avxerr = np.dot(aliveavg, (np.mean(exiterrs, axis=0) / aliveavg)) / np.sum(aliveavg)

	# Calculate saa
	saa_part1 = aliveavg @ aliveavg / np.sum(aliveavg)
	saa_part2 = CAaliveavg @ simavg[:, 3] / np.sum(CAaliveavg)
	saa_part3 = np.dot(aliveavg, simavg[:, 4:8]) / np.sum(aliveavg)
	saa_part4 = np.dot(ialiveavg, simavg[:, 8:10]) / np.sum(ialiveavg)
	saa_part5 = np.dot(aliveavg, simavg[:, 10]) / np.sum(aliveavg)

	# Concatenate all parts to form saa
	saa = np.hstack([saa_part1, saa_part2, saa_part3, saa_part4, saa_part5])

	# Print all years
	print("All years:", saa)

	# Print unique optimal intermediate Goods/Capital ratio
	optimal_ratio = np.unique(optNK, axis=0)
	print("Optimal intermediate Goods/Capital ratio:", optimal_ratio)

	if prnres > 1:
		# Function to calculate mean along columns
		def meanc(arr):
			return np.mean(arr, axis=0)

		# Calculate simavg
		simavg = meanc(assetSim * alivesim2) / aliveavg
		simavg = np.column_stack([
			simavg,
			meanc(debtSim * alivesim2) / aliveavg,
			meanc(outputSim * alivesim2) / aliveavg,
			meanc(totKSim * alivesim2) / aliveavg,
			meanc(expenseSim * alivesim2) / aliveavg
		])

		# Calculate gam_avg and ag2_avg
		gam_avg = meanc(hetgam.reshape(-1,1) * alivesim2) / aliveavg
		ag2_avg = meanc(hetag2.reshape(-1,1) * alivesim2) / aliveavg

		# Calculate aggTFP
		aggTFP = simavg[:, 2] / ((simavg[:, 3] ** gam_avg) * ((simavg[:, 4] - fcost + nshft) ** ag2_avg))

		# Update simavg
		simavg = np.column_stack([aliveavg.reshape(-1, 1), aggTFP.reshape(-1, 1), simavg])
		simavg = np.column_stack([
			simavg,
			meanc(profitSim * alivesim2) / aliveavg,
			meanc(cashSim * cshalivesim2) / cshaliveavg,
			meanc(divSim * divalivesim2) / divaliveavg,
			meanc(GInvSim * ialivesim2) / ialiveavg,
			meanc(intRateSim * alivesim2) / aliveavg,
			meanc(optKSim * alivesim2) / aliveavg
		])

		# Print yearly statistics
		print("Year frac alive Agg TFP Assets Debt Output Capital")
		print("Expenses Profits Cash Dividends Gr. Inv. Int rate optCap")
		seqa = np.arange(firstyr, firstyr + timespan).reshape(-1, 1)
		simavg = np.column_stack([seqa, simavg])

		# # Calculate saa
		# saa = np.column_stack([
		# 	np.dot(aliveavg, simavg[:, :8]) / np.sum(aliveavg),
		# 	np.dot(cshaliveavg, simavg[:, 8]) / np.sum(cshaliveavg),
		# 	np.dot(divaliveavg, simavg[:, 9]) / np.sum(divaliveavg),
		# 	np.dot(ialiveavg, simavg[:, 10]) / np.sum(ialiveavg),
		# 	np.dot(aliveavg, simavg[:, 11:13]) / np.sum(aliveavg)
		# ])
		#
		# # Print all years
		# print("All years", saa)

		# Prepare simdata0
		simdata0 = np.column_stack([
			np.arange(1, numsims + 1).reshape(-1, 1),
			IDsim,
			FEshks,
			ftype_sim
		])

		# Initialize mvc1 and imvc1
		mvc1 = -999 * (1 - aliveSim)
		imvc1 = -999 * (1 - ialiveSim)

		# Prepare simdata
		simdata = np.vstack([
			np.column_stack([np.ones((numsims, 1)), simdata0, (totKSim * aliveSim + mvc1)]),
			np.column_stack([2 * np.ones((numsims, 1)), simdata0, ((expenseSim - fcost) * aliveSim + mvc1)]),
			np.column_stack([3 * np.ones((numsims, 1)), simdata0, (outputSim * aliveSim + mvc1)]),
			np.column_stack(
				[4 * np.ones((numsims, 1)), simdata0, (np.exp(FEshks) * ZvalSim * aliveSim + mvc1)]),
			np.column_stack([5 * np.ones((numsims, 1)), simdata0, (assetSim * aliveSim + mvc1)]),
			np.column_stack([6 * np.ones((numsims, 1)), simdata0, (debtSim * aliveSim + mvc1)]),
			np.column_stack([7 * np.ones((numsims, 1)), simdata0, (GInvSim * ialiveSim + imvc1)]),
			np.column_stack([8 * np.ones((numsims, 1)), simdata0, (NInvSim * ialiveSim + imvc1)]),
			np.column_stack([9 * np.ones((numsims, 1)), simdata0, (ageSim * aliveSim + mvc1)])
		])

		# getcorrs(totKSim, YKratioSim, NKratioSim, GIKratioSim, DAratioSim, CAratioSim,
		# 		 DVKratioSim, dvgSim, eqinjSim, DVFEratioSim, aliveSim, ialiveSim, aliveSim,
		# 		 DVKalivesim, dvgaliveSim, divaliveSim, simwgts)

		np.savetxt('iofiles/simdata.txt', simdata)

	return (ageSim, assetSim, cashSim, debtSim, divSim, dvgSim, eqinjSim, goteqiSim, equitySim, expenseSim, fracRPSim,
			intRateSim, liqDecSim, NKratioSim, outputSim, totKSim, ZvalSim, aliveSim, ialiveSim, dvgaliveSim,
			fwdalivesim, divaliveSim, DVKalivesim, cshaliveSim, CAaliveSim, exiterrs, deprSim, NInvSim,
			GInvSim, DAratioSim, CAratioSim, YKratioSim, NIKratioSim, GIKratioSim, DVKratioSim,
			DVKratioSim, profitSim, netWorthSim, avxerr, ftype_sim)


def simprofs(FType, timespan, FSstate, checktie, farmsize, cht_sim,
			 aliveSim, ialiveSim, dvgaliveSim, divaliveSim, CAaliveSim,
			 totKSim, divSim, dvgSim, DAratioSim, NKratioSim, GIKratioSim,
			 CAratioSim, YKratioSim, simwgts, obsmat, dumswgts, prngrph):
	# Local variables
	sorttype = 0  # technology sort

	if isinstance(FType, float) or isinstance(FType, int):
		FType = np.array([FType])

	if isinstance(FSstate, float) or isinstance(FSstate, int):
		FSstate = np.array([FSstate])

	FSwgts = aliveSim * simwgts
	FSwgts = np.mean(FSwgts.T, axis=0) / np.mean(aliveSim.T, axis=0)  # farm averages

	if FType.shape[0] < obsmat.shape[0]:  # sort farms by size; do it here
		sorttype = sizevar
		if np.max(FSstate) > 0:
			FSgroups = FSstate.shape[0] + 1
			if wgtdsplit == 0:
				FSqnts, FScounts, FType = FSquant(farmsize, dumswgts[:, 0], FSstate, checktie)
			else:
				FSqnts, FScounts, FType = FSquant(farmsize, FSwgts, FSstate, checktie)
		else:
			FSgroups = 1
			FType = np.ones((farmsize.shape[0], 1))
	else:
		FSgroups = np.unique(FType, axis=0).shape[0]

	# Quantiles calculation
	tkqntsim, quantcnts = getqunts(cht_sim, FType, totKSim, aliveSim, quants_lv, timespan, simwgts)
	divqntsim, quantcnts = getqunts(cht_sim, FType, divSim, divaliveSim, quants_lv, timespan, simwgts)
	DAqntsim, quantcnts = getqunts(cht_sim, FType, DAratioSim, aliveSim, quants_rt, timespan, simwgts)
	nkqntsim, quantcnts = getqunts(cht_sim, FType, NKratioSim, aliveSim, quants_rt, timespan, simwgts)
	gikqntsim, quantcnts = getqunts(cht_sim, FType, GIKratioSim, ialiveSim, quants_rt, timespan, simwgts)
	CAqntsim, quantcnts = getqunts(cht_sim, FType, CAratioSim, CAaliveSim, quants_rt, timespan, simwgts)
	ykqntsim, quantcnts = getqunts(cht_sim, FType, YKratioSim, aliveSim, quants_rt, timespan, simwgts)
	dvgqntsim, quantcnts = getqunts(cht_sim, FType, dvgSim, dvgaliveSim, quants_rt, timespan, simwgts)
	obsavgsim, quantcnts = getqunts(cht_sim, FType, aliveSim, dumswgts, 0, timespan, simwgts)

	# # Graph matrix plotting
	# grphmtx(tkqntsim, 1, 1, quants_lv, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(divqntsim, 8, 1, quants_lv, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(DAqntsim, 12, 1, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(nkqntsim, 13, 1, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(gikqntsim, 14, 1, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(CAqntsim, 16, 1, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(ykqntsim, 17, 1, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(dvgqntsim, 18, 1, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	#
	# if prngrph == 1:
	# 	makgrph2(quants_lv, FSgroups, chrtnum, 1, sorttype)
	# 	makgrph2(quants_lv, FSgroups, chrtnum, 8, sorttype)
	# 	makgrph2(quants_rt, FSgroups, chrtnum, 11, sorttype)
	# 	makgrph2(quants_rt, FSgroups, chrtnum, 12, sorttype)
	# 	makgrph2(quants_rt, FSgroups, chrtnum, 13, sorttype)
	# 	makgrph2(quants_rt, FSgroups, chrtnum, 14, sorttype)
	# 	makgrph2(quants_rt, FSgroups, chrtnum, 16, sorttype)
	# 	makgrph2(quants_rt, FSgroups, chrtnum, 17, sorttype)
	# 	makgrph2(quants_rt, FSgroups, chrtnum, 18, sorttype)

	return tkqntsim, DAqntsim, nkqntsim, gikqntsim, CAqntsim, ykqntsim, divqntsim, dvgqntsim, obsavgsim


def wvec2(countadj, pdim):
	"""
	Creates a weight vector based on the count adjustment and dimensional orders.
	:param countadj: Count adjustment factor.
	:param pdim: Dimensional orders of the input.
	:return: Weight vector.
	"""
	# Compute the product of all dimensions
	total_elements = np.prod(pdim)
	# Compute the product of the specified slice of dimensions
	specified_elements = np.prod(pdim[2:4])

	# Create the vector
	vec2 = np.ones(total_elements).reshape(-1,1) * (countadj.reshape(-1,1) * np.ones(specified_elements)).reshape(-1,1)
	return vec2


def makewgtvec(tkqnt, divqnt, dvgqnt, DAqnt, CAqnt, nkqnt, gikqnt, ykqnt, obsavg, countadj):

	pdim = tkqnt.shape
	tkwgts = momwgts[0] * wvec2(countadj, pdim)

	if divmmts == 1:
		pdim = divqnt.shape
		divwgts = momwgts[1] * wvec2(countadj, pdim)
	elif divmmts == 2:
		pdim = dvgqnt.shape
		divwgts = momwgts[1] * wvec2(countadj, pdim)
	else:
		divwgts = np.array([])  # Empty array if no conditions are met

	pdim = DAqnt.shape
	DAwgts = momwgts[2] * wvec2(countadj, pdim)

	pdim = CAqnt.shape
	CAwgts = momwgts[3] * wvec2(countadj, pdim)

	pdim = nkqnt.shape
	nkwgts = momwgts[4] * wvec2(countadj, pdim)

	pdim = gikqnt.shape
	gikwgts = momwgts[5] * wvec2(countadj, pdim)

	pdim = ykqnt.shape
	ykwgts = momwgts[6] * wvec2(countadj, pdim)

	pdim = obsavg.shape
	obswgts = momwgts[7] * wvec2(countadj, pdim)

	# Concatenate all weight vectors
	wgtvec = np.concatenate([tkwgts, divwgts, DAwgts, CAwgts, nkwgts, gikwgts, ykwgts, obswgts])

	return wgtvec


def onerun(parmvec, betamax, linprefs, nobeq, w_0, bigR, numFTypes, inadaU, nonshft, noDScost, nofcost,
		  	nocolcnst, prnres, noReneg, finparms0, idioshks, randrows,
		   rloutput, totcap, intgoods, obsmat,
		   farmtype, av_cows, famsize, datawgts, chrttype, iobsmat,
		   dvgobsmat, dividends, divgrowth, LTKratio, debtasst, nkratio,
		   gikratio, CAratio, ykratio, dumdwgts, numsims):
	# Initialize placeholders and variables
	zero_vec = np.zeros_like(parmvec)
	fix_vals = np.ones_like(parmvec)  # Replace with actual fixed values
	parm_trans = 1  # Assuming some transformation flag, replace if needed

	all_parms = parmvec * zero_vec + fix_vals * (1 - zero_vec)

	if parm_trans == 1:
		pref_parms, fin_parms, gam, ag2, nshft, fcost = makepvecs(all_parms, betamax, linprefs, nobeq, w_0, bigR, numFTypes, inadaU, nonshft, noDScost, nofcost,
																  nocolcnst, prnres, noReneg, finparms0)
	else:
		pref_parms, fin_parms, gam, ag2, nshft, fcost = makepvecs2(all_parms, linprefs, w_0, bigR, inadaU, nonshft, noDScost, nofcost,
			nocolcnst, prnres, noReneg, finparms0)

	# Dataset preparation
	dataset = datasetup(gam, ag2, nshft, fcost, rloutput, totcap, intgoods, obsmat,
														   farmtype, av_cows, famsize, datawgts, chrttype, iobsmat,
														   dvgobsmat, dividends, divgrowth, LTKratio, debtasst, nkratio,
														   gikratio, CAratio, ykratio, dumdwgts)
	(TFPaggshks, TFP_FE, TFPaggeffs, tkqntdat, DAqntdat, CAqntdat, nkqntdat,
	 gikqntdat, ykqntdat, divqntdat, dvgqntdat, obsavgdat, tkqcnts, divqcnts, dvgqcnts,
	 std_zi, zvec, fevec, k_0, optNK, optKdat, countadj) = dataset

	# Handle near-zero values in `dvgqntdat`
	tinydvg = np.abs(dvgqntdat) < 0.1
	dvgqntdat = dvgqntdat * (1 - tinydvg) + 0.1 * tinydvg * (2 * (dvgqntdat > 0) - 1)

	# Handle missing data
	missmomvec, obsmmtind = missingvec(tkqntdat, divqntdat, dvgqntdat, DAqntdat, CAqntdat,
									   nkqntdat, gikqntdat, ykqntdat, obsavgdat, tkqcnts,
									   divqcnts, dvgqcnts)
	alldmoms = makemomvec(tkqntdat, divqntdat, dvgqntdat, DAqntdat, CAqntdat, nkqntdat,
						  gikqntdat, ykqntdat, obsavgdat)

	aggshks = np.hstack([0, TFPaggshks, 0])
	zshks = aggshks.T + std_zi * idioshks  # Ensure `idioshks` is defined
	zshks = zshks.flatten()
	feshks = TFP_FE[randrows]  # Ensure `randrows` is defined
	k_0 = k_0[randrows]  # Ensure `randrows` is defined
	optNK = optNK[randrows]  # Ensure `randrows` is defined

	# Save intermediate results, replace `save_path` with actual save logic if needed
	# save_results(iopath, job, pref_parms, fin_parms, zvec, fevec, zshks, feshks)

	### TODO: HERE THEY CALL THE C PROGRAM
	## Add the python version of the C
	# execret = exec(rulecall, "")  # Ensure `exec_rulecall` is defined

	## WE NEED TO WAIT UNTIL THIS IS DONE AS WE NEED FILES FROM THE C CODE
	# Load simulations
	(ageSim, assetSim, cashSim, debtSim, divSim, dvgSim, eqinjSim, goteqiSim, equitySim, expenseSim, fracRPSim,
	 intRateSim, liqDecSim, NKratioSim, outputSim, totKSim, ZvalSim, aliveSim, ialiveSim, dvgaliveSim,
	 fwdalivesim, divaliveSim, DVKalivesim, cshaliveSim, CAaliveSim, exiterrs, deprSim, NInvSim,
	 GInvSim, DAratioSim, CAratioSim, YKratioSim, NIKratioSim, GIKratioSim, DVKratioSim,
	 DVKratioSim, profitSim, netWorthSim, avxerr, ftype_sim) =	loadSims(fcost, gam, ag2, nshft, k_0, optNK, timespan, numsims,
																cashlag, tfplag, divlag, divbasemin, bigG,
																gkE, dlt, rdgE, mvcode, feshks, prnres)

	# Load files
	cht_sim = np.loadtxt('iofiles/cht_sim.txt')
	cht_sim = cht_sim.reshape(-1,1)

	if sizevar == 1:
		fsSim = feshks
	elif sizevar == 2:
		fsSim = av_cows[randrows] / famsize[randrows]  # Ensure `av_cows` and `famsize` are defined

	if GMMsort == 0:
		profsort = 0
	else:
		profsort = ftype_sim

	sim_profiles = simprofs(profsort, timespan, FSstate, checktie, fsSim, cht_sim, aliveSim,
							ialiveSim, dvgaliveSim, divaliveSim, CAaliveSim, totKSim, divSim,
							dvgSim, DAratioSim, NKratioSim, GIKratioSim, CAratioSim, YKratioSim,
							simwgts, obsmat, dumswgts, prngrph)

	(tkqntsim, DAqntsim, nkqntsim, gikqntsim, CAqntsim, ykqntsim, divqntsim, dvgqntSim, obsavgsim) = sim_profiles

	allsmoms = makemomvec(tkqntsim, divqntsim, dvgqntSim, DAqntsim, CAqntsim, nkqntsim,
						  gikqntsim, ykqntsim, obsavgsim)

	wgtvec = makewgtvec(tkqntsim, divqntsim, dvgqntSim, DAqntsim, CAqntsim, nkqntsim,
						gikqntsim, ykqntsim, obsavgsim, countadj)
	datamoms = alldmoms[(1-missmomvec).astype(bool)]
	simmoms = allsmoms[(1-missmomvec).astype(bool)]

	wgtvec = wgtvec[(1-missmomvec).astype(bool)] / datamoms.reshape(-1,1)
	rn = wgtvec.shape[0]
	wgtmtx = np.eye(rn) * (wgtvec ** 2)

	diff = datamoms - simmoms
	criter = diff.T @ wgtmtx @ diff

	# Save final results
	# save_results(iopath, datamoms, simmoms, diff, criter, wgtmtx, alldmoms, allsmoms, missmomvec, obsmmtind)

	lbl = ["Total Capital       ", "Dividends           ", "Debt/Assets         ",
		   "Cash/Assets         ", "Int. Goods/Capital  ", "Gross Invst/Capital ",
		   "Output/Capital      ", "Exit Errors         ", "Total               "]

	if divmmts == 0:
		lbl = [lbl[0]] + lbl[2:9]
		obsmmtind = obsmmtind[[0] + list(range(2, 8)), :]

	for iCrit in range(len(lbl) - 1):
		subcrit = diff[obsmmtind[iCrit, 0]:obsmmtind[iCrit, 1]]
		subwgt = wgtmtx[obsmmtind[iCrit, 0]:obsmmtind[iCrit, 1], obsmmtind[iCrit, 0]:obsmmtind[iCrit, 1]]
		subcrit = subcrit.T @ subwgt @ subcrit
		print(lbl[iCrit], subcrit)

	print(lbl[-1], criter)

	return criter


def removeFE(datamat_j, obsmat_j):
	# Calculate the sum of observations for each individual
	obscounts = np.sum(obsmat_j, axis=1)

	# Calculate the sum of observed data for each individual
	obssums = np.sum(datamat_j * obsmat_j, axis=1)

	# Calculate fixed effects values
	FEvals = obssums / obscounts

	# Extend fixed effects values to the shape of the data matrix
	FEmtx = obsmat_j * FEvals[:, np.newaxis]

	# Remove fixed effects from the data matrix
	datamat_dm = datamat_j - FEmtx

	return datamat_dm, FEvals


def loaddat(nyrsdat, lsdadj, cashtfp, chrtnum, chrtstep, sizescreen, wgtddata):
	print("THE DATA SPEAKS!")

	datapath = r"C:\Users\Simon\Downloads\Jones_Pratap_AER_2017-0370_Archive\Jones_Pratap_AER_2017-0370_Archive\estimation_fake\data\Full_Sample"

	numfarms = 363

	datstr = os.path.join(datapath, "fake_av_famsize.txt")
	famsize = np.loadtxt(datstr, ndmin=2)

	datstr = os.path.join(datapath, "cpi.txt")
	cpi = np.loadtxt(datstr, ndmin=2)
	cpi_rscl = 1

	if wgtddata == 0:
		datawgts = np.ones((numfarms, nyrsdat))
	else:
		datstr = os.path.join(datapath, "fake_weights.txt")
		datawgts = np.loadtxt(datstr, ndmin=2)
		obsmat = (datawgts > -99)
		datawgts = datawgts * obsmat

	# Load own capital data
	datstr_own_capital = os.path.join(datapath, "fake_own_capital.txt")
	owncap = np.loadtxt(datstr_own_capital)

	# Initialize obsmat
	obsmat = (owncap[:, 1:nyrsdat + 1] == mvcode)
	obsmat = 1 - obsmat

	# Process own capital data
	owncap = cpi_rscl * owncap[:, 1:nyrsdat + 1] / famsize

	# Load leased capital expense data
	if cashtfp == 1:
		datstr_lease_exp = os.path.join(datapath, "fake_lease_exp_cash.txt")
	else:
		datstr_lease_exp = os.path.join(datapath, "fake_lease_exp.txt")
	lsdex = np.loadtxt(datstr_lease_exp)

	# Process leased capital expense data
	lsdex = cpi_rscl * lsdex[:, 1:nyrsdat + 1] / famsize

	# Load leased capital data
	if cashtfp == 1:
		datstr_leased_capital = os.path.join(datapath, "fake_leased_capital_cash.txt")
	else:
		datstr_leased_capital = os.path.join(datapath, "fake_leased_capital.txt")
	lsdcap = np.loadtxt(datstr_leased_capital)

	# Process leased capital data
	lsdcap = cpi_rscl * lsdcap[:, 1:nyrsdat + 1] / famsize

	# Calculate total capital and leverage-to-total-capital ratio
	totcap = owncap + lsdcap
	LTKratio = lsdcap / totcap

	# Load total asset data
	datstr_totasset = os.path.join(datapath, "fake_totasset_beg.txt")
	totasst = np.loadtxt(datstr_totasset)
	totasst = cpi_rscl * totasst[:, 1:nyrsdat + 1] / famsize

	# Calculate cash
	cash = totasst - owncap  # No leased capital at this point

	# Load total liability data
	datstr_totliab = os.path.join(datapath, "fake_totliab_beg.txt")
	totliab = np.loadtxt(datstr_totliab)
	totliab = cpi_rscl * totliab[:, 1:nyrsdat + 1] / famsize

	# Treat leased capital as debt-funded if lsdadj[1] == 1
	if lsdadj[0] == 1:
		totasst = totasst + lsdcap
		totliab = totliab + lsdcap

	# Calculate debt-to-asset ratio, equity, and cash-to-asset ratio
	debtasst = totliab / totasst
	equity = totasst - totliab
	CAratio = cash / totasst

	# Load intermediate goods data
	if cashtfp == 1:
		datstr_intgoods = os.path.join(datapath, "fake_n_exp_cash.txt")
	else:
		datstr_intgoods = os.path.join(datapath, "fake_n_exp.txt")

	intgoods = np.loadtxt(datstr_intgoods)
	intgoods = cpi_rscl * intgoods[:, 1:nyrsdat + 1] / famsize

	# Calculate net capital ratio
	nkratio = intgoods / totcap

	# Load output data
	if cashtfp == 1:
		datstr_rloutput = os.path.join(datapath, "fake_output_cash.txt")
	else:
		datstr_rloutput = os.path.join(datapath, "fake_output.txt")

	rloutput = np.loadtxt(datstr_rloutput)
	rloutput = cpi_rscl * rloutput[:, 1:nyrsdat + 1] / famsize

	# Calculate yield-to-capital ratio
	ykratio = rloutput / totcap

	# Initializing invtyinv
	invtyinv = mvcode * np.ones((numfarms, nyrsdat))

	# Loading cashflow data
	datstr = os.path.join(datapath, "fake_cash_flow.txt")
	cashflow = np.loadtxt(datstr)
	cashflow = cashflow[:, 1:nyrsdat + 1] * cpi_rscl / famsize

	# Loading eqinject data

	datstr = os.path.join(datapath, "fake_equity_inject.txt")
	eqinject = np.loadtxt(datstr)
	eqinject = eqinject[:, 1:nyrsdat + 1] * cpi_rscl / famsize
	goteqinj = (eqinject > 0) * obsmat + mvcode * (1 - obsmat)

	# Loading dividends data
	datstr = os.path.join(datapath, "fake_div2.txt")
	dividends = np.loadtxt(datstr)
	dividends = dividends[:, 1:nyrsdat + 1] * cpi_rscl / famsize

	if sepeqinjt == 0:
		dividends = dividends-eqinject

	# Calculating DVKratio
	DVKratio = dividends / owncap

	# Calculating fixeddiv
	fixeddiv = np.abs(dividends[:, :nyrsdat-1]) > divbasemin
	divsign = 1 - 2 * (dividends[:, :nyrsdat-1] < 0)
	fixeddiv = dividends[:, :nyrsdat-1] * fixeddiv + divbasemin * (1 - fixeddiv) * divsign

	# Calculating divgrowth
	divgrowth = dividends[:, 1:nyrsdat] / fixeddiv
	divgrowth = np.hstack((np.zeros((numfarms, 1)), divgrowth))

	# Creating dvgobsmat
	# Construct a column vector of zeros
	zeros_col = np.zeros((numfarms, 1))

	# Select a subset of obsmat excluding the last column
	obsmat_subset = obsmat[:, :nyrsdat - 1]

	# Concatenate zeros_col with obsmat_subset
	concatenated_matrix = np.hstack((zeros_col, obsmat_subset))

	# Perform element-wise multiplication
	dvgobsmat = obsmat * concatenated_matrix

	# Now compare dividends to farm-specific means
	DVFEratio, DVFE = removeFE(abs(dividends), obsmat)
	DVFEratio = dividends / DVFE.reshape(-1,1)

	# Load data
	datstr = os.path.join(datapath, "fake_gross_inv_own.txt")
	grossinv = np.loadtxt(datstr)
	grossinv = grossinv[:, 1:nyrsdat + 1] * cpi_rscl / famsize

	if cashtfp == 1:
		datstr = os.path.join(datapath, "fake_gross_inv_leased_cash.txt")
	else:
		datstr = os.path.join(datapath, "fake_gross_inv_leased.txt")
	grlsdinv = np.loadtxt(datstr)
	grlsdinv = grlsdinv[:, 1:nyrsdat + 1]
	iobsmat = (grlsdinv == mvcode)
	iobsmat = 1-iobsmat
	grlsdinv = grlsdinv * cpi_rscl / famsize

	# Treat changes in leased capital as investment
	if lsdadj[1] == 1:
		grossinv += grlsdinv
		gikratio = grossinv / totcap
	else:
		iobsmat = obsmat
		gikratio = grossinv / owncap

	datstr = os.path.join(datapath, "fake_net_inv_own.txt")
	netinv = np.loadtxt(datstr)
	netinv = netinv[:, 1:nyrsdat + 1] * cpi_rscl / famsize

	if cashtfp == 1:
		datstr = os.path.join(datapath, "fake_net_inv_leased_cash.txt")
	else:
		datstr = os.path.join(datapath, "fake_net_inv_leased.txt")

	ntlsdinv = np.loadtxt(datstr)
	ntlsdinv = ntlsdinv[:, 1:nyrsdat + 1] * cpi_rscl / famsize

	if lsdadj[1] == 1:
		netinv += ntlsdinv
		nikratio = netinv / totcap
	else:
		nikratio = netinv / owncap

	# //  Impute assets using model's budget constraints
	newasst = 1 - obsmat[:, :nyrsdat - 1]
	totasst2 = totasst[:, 0].reshape(-1, 1)
	i = 0
	while i < nyrsdat - 1:
		ta2 = totasst[:, i] - (dlt - gkE) * totcap[:, i] + rloutput[:, i] - intgoods[:, i] - dividends[:, i] \
			  + totliab[:, i + 1] / (1 + r_rf) - totliab[:, i]
		ta2 = newasst[:, i] * totasst[:, i + 1] + (1 - newasst[:, i]) * ta2 / bigG
		totasst2 = np.concatenate((totasst2, ta2.reshape(-1, 1)), axis=1)
		i += 1

	datstr = os.path.join(datapath, "fake_init_age.txt")
	init_age = np.loadtxt(datstr).reshape(-1, 1)

	datstr = os.path.join(datapath, "fake_init_yr.txt")
	init_yr = np.loadtxt(datstr).reshape(-1, 1)

	datstr = os.path.join(datapath, "fake_av_cows.txt")
	av_cows = np.loadtxt(datstr).reshape(-1, 1)

	datstr = os.path.join(datapath, "fake_av_milk_type.txt")
	milktype = np.loadtxt(datstr).reshape(-1, 1)

	cowsperop = av_cows / famsize
	herdtype = 1 + (cowsperop > np.median(cowsperop))
	print("Median cows per operator =", np.median(cowsperop))

	if techsort == 0:
		farmtype = herdtype
	else:
		farmtype = milktype

	IDs = np.arange(1, numfarms + 1)
	age_2001 = init_age - init_yr + 1
	yrseq = np.arange(0, nyrsdat)
	age_all = age_2001 + yrseq.reshape(1, -1)

	too_young = np.zeros_like(age_all)
	too_old = age_all > retage
	badage = (too_young+too_old) > 0
	obsmat *= 1 - badage
	iobsmat *= 1 - badage

	mvc2 = (1 - obsmat) * mvcode
	imvc2 = (1 - iobsmat) * mvcode
	dvgmvc2 = (1 - dvgobsmat) * mvcode

	owncap = obsmat * owncap + mvc2
	lsdcap = obsmat * lsdcap + mvc2
	totcap = obsmat * totcap + mvc2
	LTKratio = obsmat * LTKratio + mvc2
	totasst = obsmat * totasst + mvc2
	totliab = obsmat * totliab + mvc2
	equity = obsmat * equity + mvc2
	cash = obsmat * cash + mvc2
	intgoods = obsmat * intgoods + mvc2
	CAratio = obsmat * CAratio + mvc2
	debtasst = obsmat * debtasst + mvc2
	nkratio = obsmat * nkratio + mvc2
	rloutput = obsmat * rloutput + mvc2
	ykratio = obsmat * ykratio + mvc2
	cashflow = obsmat * cashflow + mvc2
	totasst2 = obsmat * totasst2 + mvc2
	dividends = obsmat * dividends + mvc2
	DVKratio = obsmat * DVKratio + mvc2
	DVFEratio = obsmat * DVFEratio + mvc2
	divgrowth = dvgobsmat * divgrowth + dvgmvc2
	eqinject = obsmat * eqinject + mvc2
	goteqinj = obsmat * goteqinj + mvc2
	grossinv = iobsmat * grossinv + imvc2
	gikratio = iobsmat * gikratio + imvc2
	netinv = iobsmat * netinv + imvc2
	nikratio = iobsmat * nikratio + imvc2
	invtyinv = obsmat * invtyinv + mvc2

	# //  Remove observations not in old, truncated dataset
	subsmpl = (av_cows > sizescreen[0]) & (av_cows <= sizescreen[1])
	subsmpl = np.where(subsmpl)[0]

	if len(subsmpl) < numfarms:
		numfarms = len(subsmpl)
		IDs = IDs[subsmpl]
		obsmat = obsmat[subsmpl]
		owncap = owncap[subsmpl]
		lsdcap = lsdcap[subsmpl]
		totcap = totcap[subsmpl]
		LTKratio = LTKratio[subsmpl]
		totasst = totasst[subsmpl]
		totliab = totliab[subsmpl]
		equity = equity[subsmpl]
		cash = cash[subsmpl]
		CAratio = CAratio[subsmpl]
		debtasst = debtasst[subsmpl]
		intgoods = intgoods[subsmpl]
		nkratio = nkratio[subsmpl]
		rloutput = rloutput[subsmpl]
		ykratio = ykratio[subsmpl]
		cashflow = cashflow[subsmpl]
		totasst2 = totasst2[subsmpl]
		dividends = dividends[subsmpl]
		divgrowth = divgrowth[subsmpl]
		dvgobsmat = dvgobsmat[subsmpl]
		DVKratio = DVKratio[subsmpl]
		DVFEratio = DVFEratio[subsmpl]
		eqinject = eqinject[subsmpl]
		goteqinj = goteqinj[subsmpl]
		grossinv = grossinv[subsmpl]
		gikratio = gikratio[subsmpl]
		netinv = netinv[subsmpl]
		nikratio = nikratio[subsmpl]
		iobsmat = iobsmat[subsmpl]
		invtyinv = invtyinv[subsmpl]
		age_2001 = age_2001[subsmpl]
		init_yr = init_yr[subsmpl]
		init_age = init_age[subsmpl]
		av_cows = av_cows[subsmpl]
		famsize = famsize[subsmpl]
		milktype = milktype[subsmpl]
		herdtype = herdtype[subsmpl]
		farmtype = farmtype[subsmpl]
		datawgts = datawgts[subsmpl]

	cohorts = np.quantile(age_2001, np.arange(1 / numfarms, 1, chrtstep), method='interpolated_inverted_cdf') + 1
	cohorts = np.hstack((cohorts, retage + 1)).astype(int)
	cohorts[0] = 1

	chrtcnts, chrttype = getchrt(age_2001, cohorts)
	avgage = np.zeros(chrtnum)
	for i in range(chrtnum):
		avgage[i] = np.sum((chrttype == i+1).reshape(-1,1) * age_2001) / chrtcnts[i, 3]
	avgage = np.round(avgage)
	print("Average Cohort Age:")
	print(np.column_stack((np.arange(1, chrtnum + 1), avgage, chrtcnts)))

	if prnres > 0:
		print("Leasing included in investment measures ", lsdadj[1])
		print("Equity injections subtracted from divs. ", 1 - sepeqinjt)
		print("Cash Expenditures (1) or Accrual (0)    ", cashtfp)

		i = 1
		while i <= 2:
			if i == 1:
				print("Unweighted Averages")
				obsmat2 = obsmat
				iobsmat2 = iobsmat
				datawgts2 = np.ones_like(obsmat)
			elif i == 2:
				print("\nWeighted (Herd Size) Averages")
				obsmat2 = obsmat * datawgts
				iobsmat2 = iobsmat * datawgts
				datawgts2 = datawgts

			obsavg = np.mean(obsmat2, axis=0)
			iobsavg = np.mean(iobsmat2, axis=0)
			if lsdadj[1] == 1:
				iobsavg += (iobsavg == 0) * 1e-10

			datavg = np.column_stack([obsavg,
								np.mean(totasst * obsmat2, axis=0) / obsavg,
								np.mean(totliab * obsmat2, axis=0) / obsavg,
								np.mean(rloutput * obsmat2, axis=0) / obsavg,
								np.mean(totcap * obsmat2, axis=0) / obsavg,
								np.mean(intgoods * obsmat2, axis=0) / obsavg,
								np.mean((rloutput - intgoods - totcap * rdgE) * obsmat2, axis=0) / obsavg,
								np.mean(cashflow * obsmat2, axis=0) / obsavg,
								np.mean(dividends * obsmat2, axis=0) / obsavg,
								np.mean(grossinv * iobsmat2, axis=0) / iobsavg,
								np.mean(netinv * iobsmat2, axis=0) / iobsavg,
								np.mean(invtyinv * obsmat2, axis=0) / obsavg,
								np.mean(totasst * obsmat2, axis=0) / obsavg,
								np.mean(goteqinj * obsmat2, axis=0) / obsavg,
								np.mean(DVFEratio * obsmat2, axis=0) / obsavg])
			dvgmeds, ta2 = getqunts(np.ones((numfarms, 1)), np.ones((numfarms, 1)),
								  divgrowth, dvgobsmat, 0.5, nyrsdat, datawgts2)
			dvgmeds = dvgmeds.reshape(-1, 1)
			datavg = np.hstack([datavg, dvgmeds])

			# print("      Year    frac alive    Assets        Debt       Output    Capital     Expenses")
			# print("     Profits       Cash     Dividends    Gr. Inv.   Net Inv.   Invty Inv.    iAssets")
			# print("     Eq Inj.    Rel Div.    Div. Gr.")
			# Generate the sequence of years
			years = np.arange(firstyr, firstyr + timespan)

			# Combine years and datavg horizontally
			years = years.reshape(-1, 1)
			combined_data = np.column_stack((years, datavg))
			cols = ['Year', 'frac alive', 'Assets', 'Debt', 'Output', 'Capital', 'Expenses', 'Profits', 'Cash', 'Dividends',
					'Gr. Inv.',   'Net Inv.', 'Invty Inv.', 'iAssets', 'Eq Inj.', 'Rel Div.', 'Div. Gr.']
			combined_data = pd.DataFrame(combined_data, columns=cols)
			print(combined_data)

			datavg = np.concatenate([(obsavg.reshape(1,-1) @ datavg[:, :9] / np.sum(obsavg, axis=0)).flatten(),
							   (iobsavg.reshape(1,-1) @ datavg[:, 9:11] / np.sum(iobsavg, axis=0)).flatten(),
								(obsavg.reshape(1,-1) @ datavg[:, 11:15] / np.sum(obsavg, axis=0)).flatten()])

			print("  All years ", datavg, "\n")

			getcorrs(totcap, ykratio, nkratio, gikratio, debtasst, CAratio, DVKratio,
					 divgrowth, eqinject, DVFEratio, obsmat, iobsmat, obsmat, obsmat,
					 dvgobsmat, obsmat, datawgts2)

			i += 1

	# Select first observation for each farm
	iymtx0 = np.eye(nyrsdat)

	# Select the first observation for each farm based on init_yr
	iymtx = iymtx0[init_yr.flatten().astype(int) - 1, :]

	# Calculate initial total assets for each farm
	initTA = totasst * iymtx
	initTA = np.sum(initTA, axis=1)

	# Calculate initial total capital for each farm
	initK = totcap * iymtx
	initK = np.sum(initK, axis=1)

	# Calculate initial total debt for each farm
	initdebt = totliab * iymtx
	initdebt = np.sum(initdebt, axis=1)

	# Combine farm type, initial year, initial age, age in 2001, initial total assets,
	# initial total capital, and initial total debt into a single array
	initstate = np.column_stack((farmtype, init_yr, init_age, age_2001, initTA, initK, initdebt))

	print('Data is loaded')

	return (IDs, owncap, obsmat, lsdcap, totcap, LTKratio, totasst, totliab, equity, cash,
			CAratio, debtasst, intgoods, nkratio, rloutput, ykratio, cashflow, dividends,
			divgrowth, dvgobsmat, DVKratio, DVFEratio, eqinject, goteqinj, grossinv, gikratio,
			netinv, nikratio, iobsmat, age_2001, init_yr, init_age, av_cows, famsize, cohorts,
			chrttype, milktype, herdtype, farmtype, avgage, datawgts, initstate)


def initdist(IDs, farmtype, initstate, obsmat, iobsmat, dvgobsmat, isimage, cohorts, numdraws0, simyrs, init_age, datawgts):
	"""
	INITDIST:  This gives the initial distribution of assets, debt, TFP (FE),
               age, and year of initial observation
	"""
	ageindx = np.ones(len(IDs))
	if simtype == 2:
		ageindx = (init_age[:, 0] >= isimage[0]) & (initstate[:, 0] <= isimage[1])

	num0 = np.arange(1, len(IDs)+1)
	num0 = num0[ageindx.astype(bool)]
	rn = len(num0)

	print("Number of observations in initial distribution =", rn)

	if clonesims == 0:
		numdraws = numdraws0
		randrows = np.random.uniform(size=numdraws) * rn
		randrows = num0[randrows]
	elif clonesims == 1:
		sperobs = round(numdraws0 / rn)
		numdraws = rn * sperobs
		randrows = np.repeat(num0, sperobs) - 1

	IDsim = IDs[randrows]
	ftype_sim = farmtype[randrows]
	inityrs = initstate[randrows, 1]
	initages = initstate[randrows, 2]
	a01_sim = initstate[randrows, 3]
	initTA = initstate[randrows, 4]
	initK = initstate[randrows, 5]
	initdebt = initstate[randrows, 6]
	obssim = obsmat[randrows, :]
	iobssim = iobsmat[randrows, :]
	dvgobssim = dvgobsmat[randrows, :]
	simwgts = datawgts[randrows, :]

	chrtcnts, cht_sim = getchrt(a01_sim, cohorts)

	# Assuming `save path` does some kind of file saving, you can implement this as needed in your Python code
	# save path =^iopath initages, inityrs, initta, initK, initdebt, IDsim, obssim,
	#                        iobssim, dvgobssim, cht_sim, ftype_sim, simwgts;
	np.savetxt('iofiles/initages.txt', initages)
	np.savetxt('iofiles/inityrs.txt', inityrs)
	np.savetxt('iofiles/initta.txt', initTA)
	np.savetxt('iofiles/initK.txt', initK)
	np.savetxt('iofiles/initdebt.txt', initdebt)
	np.savetxt('iofiles/IDsim.txt', IDsim)
	np.savetxt('iofiles/obssim.txt', obssim)
	np.savetxt('iofiles/iobssim.txt', iobssim)
	np.savetxt('iofiles/dvgobssim.txt', dvgobssim)
	np.savetxt('iofiles/cht_sim.txt', cht_sim)
	np.savetxt('iofiles/ftype_sim.txt', ftype_sim)
	np.savetxt('iofiles/simwgts.txt', simwgts)

	idioshks = np.random.randn(numdraws, simyrs)

	return randrows, idioshks, numdraws


def FEReg(YvarMtx, obsmat_y, XvarMtx, timeseq, statacrap):
	"""
	Fixed effect regression with time dummies
		 Assumes explanatory variables are stacked vertically
		 Stata does not estimate simple FE.  Statacrap based on xtreg manual.
	"""
	nfarms = len(YvarMtx)
	gotobs = np.sum(obsmat_y, axis=1) > 0

	if np.sum(gotobs, axis=0) < nfarms:
		gotobs_indices = np.nonzero(gotobs)[0]
		YvarMtx = YvarMtx[gotobs, :]
		obsmat_y = obsmat_y[gotobs, :]
		if isinstance(XvarMtx, int) or isinstance(XvarMtx, float):
			nX = 1 / nfarms
		else:
			nX = len(XvarMtx) / nfarms

		if nX > 0.99:
			xvm2 = []
			for iX in range(nX):
				Xtemp = XvarMtx[iX * nfarms: (iX + 1) * nfarms, :]
				xvm2.append(Xtemp[gotobs, :])
			XvarMtx = np.concatenate(xvm2, axis=0)
			nfarms = len(YvarMtx)

	nfarms = len(YvarMtx)
	obsvec_y = obsmat_y.flatten()
	y_ucmean = np.mean(YvarMtx.flatten()[obsvec_y.astype(bool)])
	Y_dm, YFE = removeFE(YvarMtx, obsmat_y)
	Yvec = Y_dm.flatten()[obsvec_y.astype(bool)]

	rn = len(Yvec)
	const = np.ones(rn)
	tn = len(timeseq)
	timedums = const
	XmatFE = np.ones(nfarms)
	td_ucmean = np.mean(const)

	if tn > 1:
		timemtx = np.ones((numfarms, 1)) * timeseq
		td_ucmean = []
		XmatFE = np.array([])
		timedums = np.array([])

		iYr = 0
		tdtemp = (timemtx == timeseq[iYr])
		td_ucmean.append(np.mean(tdtemp.flatten()[obsvec_y.astype(bool)]))
		td_dm, tdFE = removeFE(tdtemp, obsmat_y)
		tdtemp = td_dm.flatten()[obsvec_y.astype(bool)]
		timedums = tdtemp
		XmatFE = tdFE

		for iYr in range(1, tn):
			tdtemp = (timemtx == timeseq[iYr])
			td_ucmean.append(np.mean(tdtemp.flatten()[obsvec_y.astype(bool)]))
			td_dm, tdFE = removeFE(tdtemp, obsmat_y)
			tdtemp = td_dm.flatten()[obsvec_y.astype(bool)]
			timedums = np.column_stack([timedums, tdtemp])
			XmatFE = np.column_stack([XmatFE, tdFE])

	Xmat = timedums
	x_ucmean = td_ucmean

	if isinstance(XvarMtx, int) or isinstance(XvarMtx, float):
		nX = 1 / nfarms
	else:
		nX = len(XvarMtx) / nfarms

	if nX > 0.99:
		x_ucmean = []
		for iX in range(nX):
			Xtemp = XvarMtx[iX * nfarms: (iX + 1) * nfarms, :]
			x_ucmean.append(np.mean(Xtemp[obsvec_y]))
			X_dm, XFE = removeFE(Xtemp, obsmat_y)
			Xtemp = X_dm[obsvec_y]
			Xmat = np.concatenate((Xmat, Xtemp), axis=1)
			XmatFE = np.concatenate((XmatFE, XFE), axis=1)

	x_ucmean = np.array(x_ucmean)

	if statacrap == 1:
		Yvec = Yvec + y_ucmean
		Xmat = Xmat + x_ucmean
	else:  	# If means are not added back in, drop constant (or equivalent)
		cn = len(Xmat)
		Xmat = Xmat[:, 1:cn]
		XmatFE = XmatFE[:, 1:cn]

	coeffs = np.linalg.inv(Xmat.T @ Xmat) @ Xmat.T @ Yvec
	resids = Yvec - Xmat @ coeffs
	std_eps = np.std(resids) * np.sqrt((rn - 1) / (rn - nfarms - coeffs.shape[0]))
	muFE = YFE - XmatFE @ coeffs  	# Want FE component of residual, not of Y itself
	if statacrap == 0:
		coeffs = np.concatenate(([y_ucmean - x_ucmean[1:] @ coeffs], coeffs))

	# Convert time dummies into zero-mean shocks
	if tn > 1:
		if statacrap == 0:
			coeffs[1:tn] = coeffs[1:tn] + coeffs[0]
		coeffs[:tn] = coeffs[:tn] - np.mean(coeffs[:tn])

	std_FE = np.std(muFE)
	muFE = muFE + y_ucmean - x_ucmean @ coeffs - np.mean(muFE)

	meanadj = ((1 - aggshkscl) * np.std(coeffs)) ** 2 + ((1 - idioshkscl) * std_eps) ** 2
	muFE = muFE + meanadj / 2
	coeffs = coeffs * aggshkscl
	resids = resids * idioshkscl
	std_eps = std_eps * idioshkscl

	sstat_ag = [np.mean(coeffs), np.median(coeffs), np.min(coeffs), np.max(coeffs), np.std(coeffs)]
	sstat_eps = [np.mean(resids), np.median(resids), np.min(resids), np.max(resids), std_eps]
	sstat_FE = [np.mean(muFE), np.median(muFE), np.min(muFE), np.max(muFE), std_FE]

	return coeffs, sstat_ag, sstat_eps, sstat_FE, muFE


def getTFP(rloutput, totcap, intgoods, obsmat, gam, ag2, nshft, fcost, statacrap, yrseq, firstyr, farmtype):
	# Local variables initialization
	igoods2 = intgoods + nshft - fcost
	hetgam = gam[farmtype.reshape(-1,).astype(int) - 1]
	hetag2 = ag2[farmtype.reshape(-1,).astype(int) - 1]

	# Estimate TFP
	estTFP = rloutput / ((totcap ** hetgam.reshape(-1, 1)) * (igoods2 ** hetag2.reshape(-1, 1)))
	estTFP = np.log(estTFP)
	estTFP = np.where(np.isnan(estTFP), 0, estTFP)

	# Perform fixed effects regression
	TFPaggshks, sstat_ag, sstat_eps, sstat_FE, TFP_FE = FEReg(estTFP, obsmat, 0, yrseq, statacrap)

	# Calculate TFP averages and effective TFP
	TFPavgs = np.mean(estTFP * obsmat, axis=0) / np.mean(obsmat, axis=0)
	TFPaggeffs = TFPaggshks + np.mean(TFP_FE, axis=0)

	# Extract standard deviations
	std_ts = sstat_ag[4]
	std_mu = sstat_eps[4]
	std_eps = sstat_FE[4]

	if prnres == 2:
		print("TFP Aggregate Values")
		print("      Year     Agg Shks    Agg Vals    Averages")
		print((yrseq + firstyr), TFPaggshks, TFPaggeffs, TFPavgs)
		print("               time effect   residual   fixed efct")
		lbl = ["Mean           ", "Median         ", "Minimum        ", "Maximum        ", "Std. Deviation "]
		for i in range(5):
			print(lbl[i], sstat_ag[i], sstat_eps[i], sstat_FE[i])

	return TFPaggshks, std_eps, std_mu, TFP_FE, std_ts, TFPaggeffs


def fbillvec(std_zi, std_za):
	zamin = -1e10

	if std_zi > 0:
		cutoff_fb = 0.042
		zipmtx, zivals, intvals = tauch(std_zi, rho_z, 4, cutoff_fb, cutoff_fb)
	else:
		zivals = np.array([0])
		zipmtx = np.array([1])

	if std_za > 0:
		cutoff_fb = 0.00973
		zanum = 8
		zapmtx, zavals, intvals = tauch(0.993951 * std_za, rho_z, zanum, cutoff_fb, cutoff_fb)

		if farmbill[1] > 0:
			cumprobs = np.hstack((norm.cdf(intvals/std_za), np.array([1])))
			bestind = np.argmin(np.abs(cumprobs - farmbill[1]), axis=0)
			zamin = intvals[bestind]
			zavals = np.concatenate(([zamin], zavals[bestind + 1:]))
			zapmtx = np.column_stack((zapmtx[:, :bestind+1], zapmtx[:, bestind + 1:]))
			zapmtx = zapmtx[bestind:zanum, :]

	else:
		zavals = 0
		zapmtx = 1

	zpmtx = np.kron(zipmtx, zapmtx)
	zvals = np.log(np.kron(np.exp(zivals), np.exp(zavals)))
	zdim = len(zvals)
	zrank = np.argsort(zvals)
	zvals = zvals[zrank]
	zpmtx = zpmtx[:, zrank][zrank, :]

	return zvals, zpmtx, zdim, zamin


def dataprofs(FType, farmsize, FSstate, timespan, datawgts, checktie, chrttype, obsmat,
				iobsmat, dvgobsmat, quants_lv, quants_rt, totcap, dividends, divgrowth,
				LTKratio, debtasst, nkratio, gikratio, CAratio, ykratio, dumdwgts):

	sorttype = 0
	FSwgts = obsmat * datawgts
	FSwgts = np.mean(FSwgts, axis=1) / np.mean(obsmat, axis=1)  # farm averages

	if isinstance(FType, float) or isinstance(FType, int):
		FType = np.array([FType])
	if isinstance(FSstate, float) or isinstance(FSstate, int):
		FSstate = np.array([FSstate])

	if len(FType) < obsmat.shape[0]:
		sorttype = sizevar
		if np.max(FSstate) > 0:
			FSgroups = FSstate.shape[0] + 1
			if wgtdsplit == 0:
				FSqnts, FScounts, FType = FSquant(farmsize, dumdwgts[:, 0], FSstate, checktie)
			else:
				FSqnts, FScounts, FType = FSquant(farmsize, FSwgts, FSstate, checktie)
		else:
			FSgroups = 1
			FType = np.ones(farmsize.shape[0])
	else:
		FSgroups = len(np.unique(FType, axis=0))

	countadj = np.ones((chrtnum, FSgroups))  # Adjust moments for cell counts
	if wgtdmmts == 1:
		if wgtddata == 0:
			# Need to redefine FSwgts to use herd size weights
			datapath = r"C:\Users\Simon\Downloads\Jones_Pratap_AER_2017-0370_Archive\Jones_Pratap_AER_2017-0370_Archive\estimation_fake\data\Full_Sample"
			datstr = os.path.join(datapath, "fake_weights.txt")
			FSwgts = np.loadtxt(datstr, ndmin=2)
			FSwgts = FSwgts * (FSwgts > -99)
			FSwgts = obsmat * FSwgts
			FSwgts = np.mean(FSwgts, axis=1) / np.mean(obsmat, axis=1)  # farm averages

		for iC in range(chrtnum):
			for iFS in range(FSgroups):
				thisgroup = (chrttype == iC+1) * (FType == iFS+1)
				mmtprob = 1 / (chrtnum * FSgroups)
				wgtdprob = np.dot(thisgroup.T, FSwgts) / np.sum(FSwgts)
				countadj[iC, iFS] = np.sqrt(wgtdprob / mmtprob)

	countadj = countadj.ravel()

	tkqntdat, tkqcnts = getqunts(chrttype, FType, totcap, obsmat, quants_lv, timespan, datawgts)
	divqntdat, divqcnts = getqunts(chrttype, FType, dividends, obsmat, quants_lv, timespan, datawgts)
	ltkqntdat, quantcnts = getqunts(chrttype, FType, LTKratio, obsmat, quants_rt, timespan, datawgts)
	DAqntdat, quantcnts = getqunts(chrttype, FType, debtasst, obsmat, quants_rt, timespan, datawgts)
	nkqntdat, quantcnts = getqunts(chrttype, FType, nkratio, obsmat, quants_rt, timespan, datawgts)
	gikqntdat, quantcnts = getqunts(chrttype, FType, gikratio, iobsmat, quants_rt, timespan, datawgts)
	CAqntdat, quantcnts = getqunts(chrttype, FType, CAratio, obsmat, quants_rt, timespan, datawgts)
	ykqntdat, quantcnts = getqunts(chrttype, FType, ykratio, obsmat, quants_rt, timespan, datawgts)
	dvgqntdat, dvgqcnts = getqunts(chrttype, FType, divgrowth, dvgobsmat, quants_rt, timespan, datawgts)
	obsavgdat, quantcnts = getqunts(chrttype, FType, obsmat, dumdwgts, 0, timespan, datawgts)

	# grphmtx(tkqntdat, 1, 0, quants_lv, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(divqntdat, 8, 0, quants_lv, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(ltkqntdat, 11, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(DAqntdat, 12, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(nkqntdat, 13, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(gikqntdat, 14, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(CAqntdat, 16, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(ykqntdat, 17, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	# grphmtx(dvgqntdat, 18, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)

	return tkqntdat, DAqntdat, CAqntdat, nkqntdat, gikqntdat, ykqntdat, divqntdat, dvgqntdat, obsavgdat, tkqcnts, divqcnts, dvgqcnts, countadj


def datasetup(gam, ag2, nshft, fcost, rloutput, totcap, intgoods, obsmat, farmtype, av_cows, famsize,
			  datawgts, chrttype, iobsmat, dvgobsmat, dividends, divgrowth, LTKratio, debtasst, nkratio, gikratio, CAratio, ykratio, dumdwgts):
	# Local variables initialization
	TFPaggshks, std_zi, std_fe, TFP_FE, std_za, TFPaggeffs = getTFP(rloutput, totcap, intgoods, obsmat, gam, ag2, nshft,
																	fcost, statacrap, yrseq, firstyr, farmtype)

	# Determine farmsize based on sizevar
	if sizevar == 1:
		farmsize = TFP_FE
	elif sizevar == 2:
		farmsize = av_cows / famsize

	# Calculate standard deviation of TFP shock
	std_z = np.sqrt(std_zi ** 2 + std_za ** 2)

	zdim = 8

	# Handle different cases based on zdim
	if zdim == 1:
		TFP_FE += (std_z ** 2) / 2
		zvals = np.array([0])
		zpmtx = np.array([1])
		std_zi = 0
		TFPaggshks = np.zeros_like(TFPaggshks)
		std_z = 0
	elif farmbill[0] == 1:
		zvals, zpmtx, zdim, zamin = fbillvec(std_zi, std_za)
		if farmbill[1] > 0:
			TFPaggshks = np.maximum(TFPaggshks.T, np.tile(zamin, (TFPaggshks.shape[1], 1)))
			TFP_FE += np.log(farmbill[2])
	else:
		zpmtx, zvals, intvals = tauch(std_z, rho_z, zdim, cutoff_z, cutoff_z)

	# Ensure the last column of zpmtx sums to 1
	if zdim > 1:
		zpmtx[:, zdim - 1] = 1 - np.sum(zpmtx[:, :zdim - 1], axis=1)

	# Print transitory shocks if prnres > 1 and zdim > 1
	if prnres > 1 and zdim > 1:
		j1, j2, j3, j4, j5, j6 = markch(zpmtx, zvals)
	else:
		print(zvals)

	# Convert zvals to exponential form
	zvals = np.exp(zvals)
	zvec = np.hstack([zdim, rho_z, std_z, zvals, zpmtx.flatten()])

	# FE STUFF
	fepmtx, fevals, intvals = tauch(std_fe, 0.0, fedim, cutoff_fe, cutoff_fe)
	feprobs = fepmtx[0,:]
	mean_fe = np.mean(TFP_FE, axis=0)
	fevals = np.exp(fevals + mean_fe)
	fevec = np.hstack([fedim, 0, std_fe, fevals, feprobs])
	print('Persisntent shocks:')
	print('Level:', fevals)
	print('Logs: ', np.log(fevals))

	# Calculate farmsize related values
	hetag2 = ag2[farmtype.reshape(-1,).astype(int) - 1]
	hetgam = gam[farmtype.reshape(-1,).astype(int) - 1]
	alp = 1 - hetag2 - hetgam
	optNK = rdgE * hetag2 / hetgam
	k_0 = ((hetgam / rdgE) ** (1 - hetag2)) * (hetag2 ** hetag2) * np.exp((std_z ** 2) / 2)
	optKdat = (k_0 * np.exp(TFP_FE)) ** (1 / alp)

	# Set profsort based on GMMsort
	if GMMsort == 0:
		profsort = 0
	else:
		profsort = farmtype

	# Data profiles calculation
	tkqntdat, DAqntdat, CAqntdat, nkqntdat, gikqntdat, ykqntdat, divqntdat, dvgqntdat, obsavgdat, \
		tkqcnts, divqcnts, dvgqcnts, countadj = dataprofs(profsort, farmsize, FSstate, timespan, datawgts,
														  checktie, chrttype, obsmat, iobsmat, dvgobsmat,
														  quants_lv, quants_rt, totcap, dividends,
														  divgrowth, LTKratio, debtasst, nkratio, gikratio,
														  CAratio, ykratio, dumdwgts)

	return TFPaggshks, TFP_FE, TFPaggeffs, tkqntdat, DAqntdat, CAqntdat, nkqntdat, gikqntdat, \
		ykqntdat, divqntdat, dvgqntdat, obsavgdat, tkqcnts, divqcnts, dvgqcnts, std_zi, zvec, \
		fevec, k_0, optNK, optKdat, countadj