import numpy as np
import os
from pathlib import Path

import pandas as pd
from scipy.stats import norm

from simcrit41 import getchrt, getcorrs, getqunts, tauch, FSquant, makepvecs
from markch import markch
from variables import *
#from babyfarm42b_cash import parmvec
import matplotlib.pyplot as plt



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

	#datapath = r"C:\Users\Simon\Downloads\Jones_Pratap_AER_2017-0370_Archive\Jones_Pratap_AER_2017-0370_Archive\estimation_fake\data\Full_Sample"
	datapath = "/Users/hjaltewallin/Code/Jones_Pratap_AER_2017-0370_Archive/estimation_fake/data/Full_Sample"
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

	# // Remove observations not in old, truncated dataset
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
	obssim = obsmat[randrows]
	iobssim = iobsmat[randrows]
	dvgobssim = dvgobsmat[randrows]
	simwgts = datawgts[randrows]

	chrtcnts, cht_sim = getchrt(a01_sim, cohorts)

	# Assuming `save path` does some kind of file saving, you can implement this as needed in your Python code
	# save path =^iopath initages, inityrs, initta, initK, initdebt, IDsim, obssim,
	#                        iobssim, dvgobssim, cht_sim, ftype_sim, simwgts;

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
			# TODO: Need to redefine FSwgts to use herd size weights
			# I think there might be some problems maintaining cross-platform compatibility if we go 2 directories deep.
			# move all of /data/Full_sample/ to just /data/?
			
			FSwgts = load_file("fake_weights.txt", subdir="data/Full_sample", ndmin=2)
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

def generate_all_summary_statistics():
	"""Outputs summary statistics in LATEX format"""

	(IDs, owncap, obsmat, lsdcap, totcap, LTKratio, totasst, totliab, equity, cash,
			CAratio, debtasst, intgoods, nkratio, rloutput, ykratio, cashflow, dividends,
			divgrowth, dvgobsmat, DVKratio, DVFEratio, eqinject, goteqinj, grossinv, gikratio,
			netinv, nikratio, iobsmat, age_2001, init_yr, init_age, av_cows, famsize, cohorts,
			chrttype, milktype, herdtype, farmtype, avgage, datawgts, initstate) = loaddat(timespan, np.array([1, 0]), 1, chrtnum, chrtstep, sizescreen, wgtddata) # du skal ikke spørge hvorfor

	# LATEX table header
	print("\\begin{tabular}{lccccc}")
	print("\\headrow{Variable & Mean & Median & Std. Dev. & Max & Min} \\\\")

	# TODO: Implement periode-kode, som angiver, hvilken periode det er i. 
	# Nemmere løsning: Vi opdeler dem på (år,værdi), og så snitter vi over værdi betinget på år (=0)

	summary_statistic_dictionary = {
								"Family size": famsize,
								"youngest operator age": init_age,
								"totals assets": totasst,
								"average age": avgage,
								"divident growth": divgrowth}

	# Loop through the dictionary and calculate statistics
	for key, element in summary_statistic_dictionary.items():
		summary_statistic = return_individual_sum_stats(element)
		# Format statistics for LATEX output
		print(f"{key} \t\t& {summary_statistic[0]:.2f} & {summary_statistic[1]:.2f} & {summary_statistic[2]:.2f} & {summary_statistic[3]:.0f} & {summary_statistic[4]:.0f} \\\\")

	# LATEX table footer
	print("\\end{tabular}")

def return_individual_sum_stats(statistic, mvcode=-99):
    """
    Calculates summary statistics, handling missing values (mvcode).

    Args:
        statistic (numpy.ndarray): The array containing data for which to calculate statistics.
        mvcode (int, optional): The code representing missing values. Defaults to -99.

    Returns:
        numpy.ndarray: An array containing the calculated statistics (mean, median, std, min, max).
    """

    # Remove observations with missing values (mvcode) before calculating statistics
    filtered_statistic = statistic[statistic != mvcode]

    # Check if any data remains after filtering
    if not filtered_statistic.any():
        raise ValueError("No valid data for calculating statistics. All values are missing (mvcode).")

    mean = np.mean(filtered_statistic)
    median = np.median(filtered_statistic)
    std = np.std(filtered_statistic)
    min = np.min(filtered_statistic)
    max = np.max(filtered_statistic)

    return np.array((mean, median, std, min, max))

def load_file(filename, subdir="iofiles", ndmin=1):
  """Loads data from a file in a subdirectory into a NumPy array of floats.

  Args:
      filename (str, optional): The name of the data file.
      subdir (str, optional): The name of the subdirectory containing the data. Defaults to "iofiles".

  Returns:
      A NumPy array containing the data from the file, or None if the file
      is not found or an error occurs during loading.
  """

  # Construct the full path to the file
  this_directory = os.path.dirname(__file__)
  full_subdir = os.path.join(this_directory, subdir)
  filepath = os.path.join(full_subdir, filename)
  
  try:
    data = np.loadtxt(filepath, dtype=float)
    return data
  except FileNotFoundError:
    print(f"Error: File '{filepath}' not found.")
    return None


# we use parmvec here to avoid problems of indeterminate scope 
# Det her er det mest komplicerede modelarbejde jeg nogensinde har lavet på studiet. 
def loadSims(parmvec, subdir="iofiles"):
	"""
	This function is run after onerun(.), and transform our outputs to outputs, that we can graph. 
	And also use to calculate SMD-criterion. So it's very useful!!!
	"""
	
	magic_constant = 24 # TODO: Find out why this is
	print("Loading simulations...")

	# We need to load a series of files here.
	ftype_sim = load_file("ftype_sim.txt", subdir) ### NOTE: GAUSS IS *NOT* CASE-sensitive. Learned that the annoying way
	feshks = load_file("ftype_sim.txt", subdir)
	# GRRRR we need to run initdist to get these files.
	#obsSim = load_file("obs")
	# obsSim, iobsSim, dvgobsSim, FEshks, IDsim, simwgts

	ageS 		= load_file("ageS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	assetsS 	= load_file("assetsS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	cashS  		= load_file("cashS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	debtS       = load_file("debtS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	divsS       = load_file("divsS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	equityS     = load_file("equityS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	expenseS    = load_file("expenseS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	fracRPS     = load_file("fracRPS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	intRateS    = load_file("intRateS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	liqDecS     = load_file("liqDecS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	outputS     = load_file("outputS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	totKS       = load_file("totKS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T
	ZvalsS      = load_file("ZvalS.txt", subdir).reshape(timespan + 1, numsims+magic_constant).T

	ageSim = ageS[:, :timespan]  # All rows, columns 0 to timespan-1 (inclusive)
	assetSim = assetsS[:, :timespan] + 1e-10  # Add small value to avoid division by zero
	cashSim = cashS[:, cashlag:timespan+cashlag]  # Shift by cashlag periods
	debtSim = debtS[:, :timespan]  # All rows, columns 0 to timespan-1 (inclusive)
	equitySim = equityS[:, :timespan]  # All rows, columns 0 to timespan-1 (inclusive)
	expenseSim = expenseS[:, :timespan]  # All rows, columns 0 to timespan-1 (inclusive)
	fracRPSim = fracRPS[:, :timespan]  # All rows, columns 0 to timespan-1 (inclusive)
	intRateSim = intRateS[:, :timespan]  # All rows, columns 0 to timespan-1 (inclusive)
	outputSim = outputS[:, :timespan]  # All rows, columns 0 to timespan-1 (inclusive)
	ZvalSim = ZvalsS[:, tfplag:timespan+tfplag]  # Shift by tfplag periods	

	prefparms, finparms,gam, ag2, nshft, fcost = makepvecs(parmvec)

	hetag2 = ag2[ftype_sim]
	hetgam = gam[ftype_sim]
	alp = 1 - hetag2 - hetgam

	(TFPaggshks, TFP_FE, TFPaggeffs, tkqntdat, DAqntdat, CAqntdat, nkqntdat, gikqntdat, 
  	 ykqntdat, divqntdat, dvgqntdat, obsavgdat, tkqcnts, divqcnts, dvgqcnts, std_zi, zvec, 
	 fevec, k_0, optNK, optKdat, countadj) = datasetup(gam,ag2,nshft,fcost)

	# Calculate optimal capital stock
	optKSim = (k_0 * np.exp(feshks))**(1.0 / alp)

	# Dividend calculations with lag
	divSim = divsS[:, 1+divlag:timespan+divlag]

	# Fixed dividend logic
	fixeddiv = np.abs(divSim[:, 1:timespan-1]) > divbasemin  # Logical comparison for fixed dividends
	divsign = 1 - 2 * (divSim[:, 1:timespan-1] < 0)  # Sign of dividends

	# Fixed dividend calculation with minimum threshold
	fixeddiv = divSim[:, 1:timespan-1] * fixeddiv + divbasemin * (1 - fixeddiv) * divsign

	# Dividend growth calculation (assuming dvgSim is desired output)
	dvgSim = divSim[:, 2:timespan] / fixeddiv  # Growth rate (avoid division by zero)
	dvgSim = np.zeros((numsims, 1))  # Assuming you want to initialize dvgSim to zeros
	dvgSim[:-1] = dvgSim[1:]  # Copy growth rates (excluding the first element)

	# Equity injection calculations
	eqinjSim = divSim.copy()  # Copy dividend for equity injection
	goteqiSim = eqinjSim > 0  # Logical comparison for positive equity injection

	# Identify alive firms (investment decision)
	aliveSim = liqDecS == 0  # Logical comparison for alive firms

	# Forward-looking alive firms (assuming timespan starts from 2)
	fwdalivesim = aliveSim[:, 2:timespan+1]

	# Net investment
	NInvSim = (totKS[:, 2:timespan+1] * fwdalivesim) * bigG / (1 + gkE) - totKS[:, 1:timespan] * aliveSim[:, 1:timespan]

	# Depreciation
	deprSim = totKS[:, 1:timespan] * aliveSim[:, 1:timespan] * dlt

	# Gross investment
	GInvSim = NInvSim + deprSim

	# Total capital stock (avoid division by zero)
	totKSim = totKS[:, 1:timespan] + 1e-10

	# Profit
	profitSim = outputSim - totKSim * rdgE - expenseSim

	# Net worth
	netWorthSim = assetSim - debtSim

	# Investment to output ratio (assuming hetag2 and expenseSim-fcost+nshft are element-wise non-zero)
	iZvalSim = outputSim / ((totKSim**hetgam) * ((expenseSim - fcost + nshft)**hetag2))

	# Liquidation decision (excluding first period)
	liqDecSim = liqDecS[:, 1:timespan]

	# Alive firms with forward-looking adjustment (excluding simulated liquidators)
	ialiveSim = aliveSim[:, 1:timespan] * fwdalivesim

	# Debt-to-value of exiting firms with lag (excluding the last period)
	dvgaliveSim = aliveSim[:, 1+divlag:timespan+divlag] * (np.zeros((numsims, 1)) != aliveSim[:, 1+divlag:timespan+divlag-1])

	# Alive firms (excluding first period)
	aliveSim = aliveSim[:, 1:timespan]

	# Exit errors (based on observed exit and simulated exit)
	exiterrs = (alivesim == 0) * obsSim  # Element-wise multiplication

	# Any exit errors in the simulation
	gotxerrs = (exiterrs * np.ones((timespan, 1)) > 0).any(axis=0)  # Check if any errors exist in any simulation

	# List of unique firm IDs
	iDlist = np.unique(IDSim)

	# Dummy variables for fixed effects regression
	iDDums = IDSim[:, None] == iDlist  # Broadcast comparison for dummies

	# Average exit errors by fixed effects
	# Assuming invpd is a function for inverting a matrix with potential singular values
	xerravg = np.linalg.pinv(iDDums.T @ iDDums) @ (iDDums.T @ (gotxerrs != exiterrs))

	# Assuming iobsSim, dvgobsSim, obsSim, cashlag, divaliveSim (potentially from previous steps) are already defined

	# Alive firms with observation adjustment
	ialiveSim = iobsSim * ialiveSim  # Element-wise multiplication

	# Debt-to-value of exiting firms with observation adjustment
	dvgaliveSim = dvgobsSim * dvgaliveSim  # Element-wise multiplication

	# Alive firms based on observations
	aliveSim = obsSim * aliveSim  # Element-wise multiplication

	# Forward-looking alive firms with observation adjustment
	fwdalivesim = obsSim * fwdalivesim  # Element-wise multiplication

	# Debt-weighted average capital stock of alive firms
	divaliveSim = divlag * fwdalivesim + (1 - divlag) * aliveSim  # Weighted average

	# Debt-value product of alive firms
	DVKalivesim = divaliveSim * aliveSim  # Element-wise multiplication

	# Cash-weighted average capital stock of alive firms
	cshaliveSim = cashlag * fwdalivesim + (1 - cashlag) * aliveSim  # Weighted average

	# Cash-value product of alive firms
	CAaliveSim = cshaliveSim * aliveSim  # Element-wise multiplication




def comparison_graph(simulated_series, real_series):
	# Given a time series of simulated and real data, where the x-axis is age, 
	# and the y-axis is given by the input, output two graphs
	pass




def datasetup(gam, ag2, nshft, fcost):
	# Generate a bunch of variables that we need!!

	# We need access to these variables so that we can calculate TFP next line (first 4 arguments)
	(IDs, owncap, obsmat, lsdcap, totcap, LTKratio, totasst, totliab, equity, cash,
			CAratio, debtasst, intgoods, nkratio, rloutput, ykratio, cashflow, dividends,
			divgrowth, dvgobsmat, DVKratio, DVFEratio, eqinject, goteqinj, grossinv, gikratio,
			netinv, nikratio, iobsmat, age_2001, init_yr, init_age, av_cows, famsize, cohorts,
			chrttype, milktype, herdtype, farmtype, avgage, datawgts, initstate) = loaddat(timespan, np.array([1, 0]), 1, chrtnum, chrtstep, sizescreen, wgtddata) # du skal ikke spørge hvorfor

	TFPaggshks,std_zi,std_fe,TFP_FE,std_za,TFPaggeffs = getTFP(rloutput,totcap,intgoods,obsmat,gam,ag2,nshft,fcost,statacrap,yrseq,firstyr,farmtype)

	if sizevar==1:
		farmsize = TFP_FE		# Jones says: Global. What does it mean?
	elif sizevar==2:
		farmsize = av_cows/famsize

	std_z     = np.sqrt(std_zi**2+std_za**2);        # @ Std deviation of TFP shock @
	if zdim==1:                               # @ no transitory uncertainty @
		TFP_FE     = TFP_FE + (std_z**2)/2
		zvals      = 0
		zpmtx      = 1
		std_zi     = 0
		TFPaggshks = 0*TFPaggshks
		std_z      = 0
	else:
		zpmtx, zvals, intvals = tauch(std_z,rho_z,zdim,cutoff_z,cutoff_z) # tauch?

	if zdim>1:
		zpmtx[:,zdim] = np.ones((zdim,1))-np.sum(zpmtx[:,1:zdim-1].T, 0)	
    	
	print("Transitory shocks:")

	if (prnres>1) and (zdim>1):
		j1, j2, j3, j4, j5, j6 = markch(zpmtx, zvals)
	else:
		print(zvals)			# I think this is right?

	zvals = np.exp(zvals)

    # zvec      = zdim|rho_z|std_z|zvals|vecr(zpmtx)
	# maybe rhoz, std_z need to be np.array(rho_z), np.array(std_z)
	zvec = np.concatenate((zdim, rho_z, std_z, zvals, np.vectorize(lambda x: x.flatten())(zpmtx)))

	# Assuming tauch is a function that generates persistent shocks

	# Generate persistent shocks using tauch
	fepmtx, fevals, intvals = tauch(std_fe, 0.0, fedim, cutoff_fe, cutoff_fe)

	# Extract transition probabilities
	feprobs = fepmtx[1, :].T  # Assuming row 1 contains transition probabilities

	# Calculate average TFP level
	mean_fe = np.mean(TFP_FE)

	# Log-transform fevals and add mean for persistent deviations
	fevals = np.exp(fevals + mean_fe)

	# Combine variables into a vector
	# I'm a little uncertain about the dimensions here, but let's look at that l8r 
	fevec = np.concatenate((fedim, [0], [std_fe], fevals, feprobs))

	print("Persistent Shocks:")
	print("Levels:", fevals)
	print("Logs :", np.log(fevals))

	# Assuming ag2, gam, rdgE, std_z, TFP_FE are already NumPy arrays

	# Extract data based on farm type
	hetag2 = ag2[farmtype]
	hetgam = gam[farmtype]

	# Calculate parameters
	alp = 1 - hetag2 - hetgam
	optNK = rdgE * hetag2 / hetgam  # Element-wise division

	# Calculate initial capital stock
	k_0 = ((hetgam / rdgE) ** (1 - hetag2)) * (hetag2**hetag2) * np.exp(std_z**2 / 2)

	# Calculate optimal capital stock
	optKdat = (k_0 * np.exp(TFP_FE))**(1.0 / alp)

	# Farm type sorting logic
	if GMMsort == 0:
		profsort = 0  # No sorting for GMM
	else:
		profsort = farmtype  # Use farm type for sorting

	# beautiful
	(tkqntdat, DAqntdat, CAqntdat, nkqntdat, gikqntdat, ykqntdat, divqntdat, dvgqntdat, 
     obsavgdat, tkqcnts, divqcnts, dvgqcnts, countadj) = dataprofs(profsort, farmsize, 
				FSstate, timespan, datawgts, checktie, chrttype,  
				obsmat, iobsmat, dvgobsmat, quants_lv, quants_rt, timespan, totcap, 
                dividends, divgrowth, LTKratio, debtasst, nkratio, gikratio, CAratio, 
                ykratio)
    
	return (TFPaggshks, TFP_FE, TFPaggeffs, tkqntdat, DAqntdat, CAqntdat, nkqntdat, gikqntdat, 
   			  ykqntdat, divqntdat, dvgqntdat, obsavgdat, tkqcnts, divqcnts, dvgqcnts, std_zi, zvec, 
     		  fevec, k_0, optNK, optKdat, countadj)




################   ENTRY POINT   ########
if __name__ == "__main__":
	print("__name__ == '__main__'")
	generate_all_summary_statistics()

	#loadSims(parmvec)




#
# if __name__ == "__main__":
# 	numfarms = 363
# 	mvcode = -99
# 	statacrap = 0
# 	divbasemin = 1  # minimum abs value of denominator in div growth calculation
# 	bigG = 1  # farm size growth
#
# 	firstyr = 2001
# 	lastyr = 2011
# 	timespan = lastyr - firstyr + 1
#
# 	gkvals = 0.035582
# 	gkpmtx = 1
# 	gkE = gkpmtx * gkvals
# 	wgtddata = 0  # 1 => Herd size weights
#
# 	r_rf = 0.04  # Risk-free ROR
# 	dlt = 0.055545  # depreciation rate
# 	techsort = 1  # 0 => sort farm tech by herd size, 1 => sort by tech type
# 	gkE = gkpmtx * gkvals
# 	gkdim = 1  # dimension of capital price shock
# 	std_gk = 0  # s.d. of capital gains shocks
# 	gkvec = np.array([gkdim, 0, std_gk, gkvals, gkpmtx])
# 	rdg = r_rf + dlt - gkvals
# 	rdgE = r_rf + dlt - gkE
#
# 	retage = 75
# 	sizescreen = [0, 1e6]
# 	chrtnum = 2
# 	chrtstep = 1 / chrtnum
#
# 	prnres = 2
# 	prngrph = 1
#
# 	sepeqinjt = 0  # 0 => treat equity injections as neg dividends; 1 => separate
#
# 	nyrsdat = 11
# 	lsdadj = [1, 0]
# 	cashtfp = 1
#
# 	# Example usage:
# 	simtype = 1  # 1 => all ages; 2 => specific age
# 	clonesims = 1  # 0 => bootstrap initial distribution; 1 => scale up
#
