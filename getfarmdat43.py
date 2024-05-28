import numpy as np
import os
from pathlib import Path


import pandas as pd
from scipy.stats import norm

from simcrit41 import getchrt, getcorrs, getqunts, tauch, FSquant, makepvecs
from markch import markch
from utility_functions import load_file, removeFE # we move this out to resolve a circular import.
from variables import *
#from babyfarm42b_cash import parmvec
import matplotlib.pyplot as plt


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
	y_ucmean = np.mean(YvarMtx.flatten()[obsvec_y.astype(bool)], axis=0)
	Y_dm, YFE = removeFE(YvarMtx, obsmat_y)
	Yvec = Y_dm.flatten()[obsvec_y.astype(bool)]

	rn = len(Yvec)
	const = np.ones(rn)
	tn = len(timeseq)
	timedums = const
	XmatFE = np.ones(nfarms)
	td_ucmean = np.mean(const, axis=0)

	if tn > 1:
		timemtx = np.ones((numfarms, 1)) * timeseq
		td_ucmean = []
		XmatFE = np.array([])
		timedums = np.array([])

		iYr = 0
		tdtemp = (timemtx == timeseq[iYr])
		td_ucmean.append(np.mean(tdtemp.flatten()[obsvec_y.astype(bool)], axis=0))
		td_dm, tdFE = removeFE(tdtemp, obsmat_y)
		tdtemp = td_dm.flatten()[obsvec_y.astype(bool)]
		timedums = tdtemp
		XmatFE = tdFE

		for iYr in range(1, tn):
			tdtemp = (timemtx == timeseq[iYr])
			td_ucmean.append(np.mean(tdtemp.flatten()[obsvec_y.astype(bool)], axis=0))
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
			x_ucmean.append(np.mean(Xtemp[obsvec_y], axis=0))
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
		coeffs[:tn] = coeffs[:tn] - np.mean(coeffs[:tn], axis=0)

	std_FE = np.std(muFE)
	muFE = muFE + y_ucmean - x_ucmean @ coeffs - np.mean(muFE, axis=0)

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
	"""
	I think this is all the stuff related to the farm bill section. Might be useful for counterfactuals?
	"""
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

def datasetup(gam, ag2, nshft, fcost, rloutput, totcap, intgoods, obsmat, 
			  farmtype, av_cows, famsize, datawgts, chrttype, iobsmat, dvgobsmat, 
			  dividends, divgrowth, LTKratio, debtasst, nkratio, gikratio,
			    CAratio, ykratio, dumdwgts, avgage):
	
	print("datasetup")

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
														  CAratio, ykratio, dumdwgts, avgage)

	return TFPaggshks, TFP_FE, TFPaggeffs, tkqntdat, DAqntdat, CAqntdat, nkqntdat, gikqntdat, \
		ykqntdat, divqntdat, dvgqntdat, obsavgdat, tkqcnts, divqcnts, dvgqcnts, std_zi, zvec, \
		fevec, k_0, optNK, optKdat, countadj

def dataprofs(FType, farmsize, FSstate, timespan, datawgts, checktie, chrttype, obsmat,
				iobsmat, dvgobsmat, quants_lv, quants_rt, totcap, dividends, divgrowth,
				LTKratio, debtasst, nkratio, gikratio, CAratio, ykratio, dumdwgts, avgage):

	print("dataprofs")

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

	print("calling grphmtx")

	grphmtx(tkqntdat, 1, 0, quants_lv, FSgroups, chrtnum, timespan, sorttype, avgage)
	grphmtx(divqntdat, 8, 0, quants_lv, FSgroups, chrtnum, timespan, sorttype)
	grphmtx(ltkqntdat, 11, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	grphmtx(DAqntdat, 12, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	grphmtx(nkqntdat, 13, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	grphmtx(gikqntdat, 14, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	grphmtx(CAqntdat, 16, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	grphmtx(ykqntdat, 17, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)
	grphmtx(dvgqntdat, 18, 0, quants_rt, FSgroups, chrtnum, timespan, sorttype)

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


def comparison_graph(simulated_series, real_series):
	# Given a time series of simulated and real data, where the x-axis is age, 
	# and the y-axis is given by the input, output two graphs
	pass

def getmatrix(a, loc):
  """
  Gets a contiguous matrix from an N-dimensional array.

  Args:
      a (N-dimensional array): The input array.
      loc (Mx1 vector): The indices into the array to locate the matrix of interest.
                          M can be N, N-1, or N-2.

  Returns:
      y (KxL matrix or 1xL matrix or scalar): The extracted matrix or scalar value.
          K is the size of the second fastest moving dimension, and L is the size
          of the fastest moving dimension.

  Raises:
      ValueError: If the dimensions of `loc` are not compatible with `a`.
  """
    
  # Handle edge cases: M = N (extract entire array)
  if loc.shape[0] == a.ndim:
    return a

  # Handle M = N-1 (extract a subarray along the last dimension)
  if loc.shape[0] == a.ndim - 1:
    sliced_array = a[tuple(loc)]
    if sliced_array.ndim == 1:
      return sliced_array.reshape(1, -1)  # Ensure row vector for consistency
    else:
      return sliced_array

  # Handle M = N-2 (extract a submatrix along the last two dimensions)
  if loc.shape[0] == a.ndim - 2:
    return a[tuple(loc)]

  # Raise an error for unsupported cases (M < N-2 or M > N)
  raise ValueError("Unsupported value for M. loc must have length N, N-1, or N-2.")


def grphmtx(dataprfs, vartype, datatype, quants_j, FSnum_j, chrtnum_j, numyrs_j, sorttype, avgage):
	"""
	WIP.
	This function shapes the data how we want it. Core to making graphs.
	We modify the args to avoid scope problems. 

	Args:
		dataprfs: (array-like) Data profiles (possibly from previous steps)
		vartype: (str) Variable type (e.g., 'Y', 'D')
		datatype: (str) Data type (e.g., 'L', 'C')
		quants_j: (array-like) Quantiles (possibly from previous step)
		FSnum_j: (int) Number of firms (or a related count)
		chrtnum_j: (int) Chart number (for potential identification)
		numyrs_j: (int) Number of years (relevant data period)
		sorttype: (str) Sorting type for data (e.g., 'asc', 'desc')
	"""

	print("------------- GRAPHMTX ------------")

	vartype_to_name = {
	1: "totK",
	2: "ownK",
	3: "TA",
	4: "D",
	5: "IG",
	6: "GI",
	7: "NI",
	8: "DV",
	9: "Y",
	10: "CF",
	11: "LTK",
	12: "DA",
	13: "NK",
	14: "GIK",
	15: "NIK",
	16: "CA",
	17: "YK",
	18: "DVG",
	}

	if vartype in vartype_to_name:
		name1 = vartype_to_name[vartype]
	else:
		# Handle potential unknown vartype values
		print(f"Warning: Unknown vartype value {vartype}")
		name1 = "Unknown vartype"

	datatype_to_suffix = {
	0: "dt",  # Data
	1: "sm",  # Simulation
	}

	if datatype in datatype_to_suffix:
		name2 = datatype_to_suffix[datatype]
	else:
		# Handle potential unknown vartype values
		print(f"Warning: Unknown datatype value {datatype}")
		name2 = "Unknown datatype"


	sorttype_to_suffix = {
	0: "TS",  # Technology sort
	1: "TFP",  # Farm size sort: TFP fixed effect
	2: "HS",  # Farm size sort: herd size
	3: "DA",  # Farm size sort: debt/asset ratio
	}

	if sorttype in sorttype_to_suffix:
		name2 += sorttype_to_suffix[sorttype]
	else:
		# Handle potential unknown vartype values
		print(f"Warning: Unknown sorttype value {sorttype}")
		name2 += "Unknown sort"

	if quants_j == 0:
		iQunt = 1
		qnum_j = quants_j.shape[0]  # Get number of rows (quantiles)
	else:
		iQunt = 0
		qnum_j = 0  # Set number of quantiles to 0	

	# Calculate age range
	_tr2 = int(np.max(avgage, axis=0) - np.min(avgage, axis = 0) + numyrs_j + 5)
	age_seq2 = np.arange(np.min(avgage, axis=0) - 2, _tr2 + 1)  # Use numpy.arange for sequence

	# Extract maturity years
	mmtyrs = dataprfs.shape[3]  # 'getorders' corresponds to np.shape. -1 bc GAUSS-indexing.

	# Create sequence of maturity years
	mmtcols = np.arange(1, mmtyrs + 1)  # Use numpy.arange for sequence

	for iQunt in range(qnum_j + 1):  # Loop through quantiles (including 0 for means)
		name3 = f"{iQunt}"  # Format quantile number as string

		# Initialize graph matrix with missing values
		gmat = np.ones((_tr2, chrtnum_j * FSnum_j)) * np.NaN  # Missing value representation

		# Track ages with observations
		gotsome = np.zeros((_tr2, 1))

		cn = 1  # Column counter

		for iChrt in range(0, chrtnum_j):  # Loop through charts
			for iFS in range(0, FSnum_j):  # Loop through firms
				if iQunt == 0:
					# Means case
					getmatrix_parameters = np.array([iChrt, iFS, 0]) # -1 til eksponent her
				else:
					# Quantile case
					getmatrix_parameters = np.array([iChrt, iFS, iQunt])
					
				# Handle missing values
				tempprf = getmatrix(dataprfs, getmatrix_parameters)  # Assuming getmatrix function
				tempprf = np.where(tempprf == mvcode, np.NaN, tempprf)

				cn += 1
				rn = int(mmtyrs + avgage[iChrt - 1] - np.min(age_seq2, axis=0))  # Calculate row indices

				# Track ages with observations
				gotsome[rn] = np.ones((mmtyrs,1)).flatten() # gotsome[rn] is shape (1,), while RHS is (11,). hmm...

		# Fill graph matrix
		gmat[rn, cn - 1] = tempprf.T  # Transpose for row-wise storage

		# Remove rows with no observations
		gmat = gmat[gotsome.flatten() == 1, :]
		ageseq3 = age_seq2[gotsome.flatten() == 1]

		# Interpolation for missing values within columns
		for col in range(1, gmat.shape[1]):
			gmat[:, col] = np.interp(ageseq3, age_seq2, gmat[:, col], where=np.isnan(gmat[:, col]))
	
	
	grphpath = "..." ### PLACEHOLDER

	# Create filenames and save graph matrices
	fnamestr = grphpath + name1 + name2 + name3
	np.save(fnamestr, gmat)

	if datatype == 1 and basecase:  # Assuming datatype and basecase are defined
	# Save for comparison graphs
		fnamestr = grphpath + name1 + name2 + "bn" + name3
		np.save(fnamestr, gmat)

def makgrph2(quants_j, FSnum_j, chrtnum_j, grphtype, sorttype):
	"""
	This is the massive Goodness-of-fit plotting function. Its a wrapper for the settings 
	for the 18 graphs that we can make. Very annoying. Uses input from grphmtx (graph matrix)

	Args:
	quants_j: (array-like) Quantiles (possibly from previous step)
	FSnum_j: (int) Number of firms (or a related count)
	chrtnum_j: (int) Chart number (for potential identification)
	grphtype: (str) Graph type (e.g., 'bar', 'line')
	sorttype: (str) Sorting type for data (e.g., 'asc', 'desc')
	"""
	# Handle grphtype argument
	
	type_map = {
	1: ("Capital", "totK"),
	2: ("Owned Capital", "ownK"),
	3: ("Total Assets", "TA"),
	4: ("Debt", "D"),
	5: ("Equity", "E"),
	6: ("Int. Goods", "IG"),
	7: ("Gross Invst", "GI"),
	8: ("Dividends", "DV"),
	9: ("Output", "Y"),
	10: ("Cashflow", "CF"),
	11: ("Leased/Total Ratio", "LTK"),
	12: ("Debt/Asset Ratio", "DA"),
	13: ("N/K Ratio", "NK"),
	14: ("GI/K Ratio", "GIK"),
	15: ("NI/K Ratio", "NIK"),
	16: ("Cash/Asset Ratio", "CA"),
	17: ("Output/Capital Ratio", "YK"),
	18: ("Dividend Growth", "DVG"),
	}

	if grphtype in type_map:
		typestr, typestr2 = type_map[grphtype]
	else:
		# Handle potential unknown grphtype values (optional)
		print(f"Warning: Unknown grphtype value {grphtype}")
		typestr, typestr2 = "Unknown", "Unknown"

	print(f"Generating graph for {typestr} ({typestr2})")

	# Handle sorttype argument
	sort_map = {
	0: ("TS", "Technology"),
	1: ("TFP", "TFP"),
	2: ("HS", "Cows"),
	3: ("DA", "D/A Ratio"),
	}

	if sorttype in sort_map:
		typestr3, titlstr2 = sort_map[sorttype]
	else:
		# Handle potential unknown sorttype values (optional)
		print(f"Warning: Unknown sorttype value {sorttype}")
		typestr3, titlstr2 = "Unknown", "Unknown"

	# Use typestr3 and titlstr2 for your graph generation logic
	print(f"Sorting by {titlstr2} ({typestr3})")

	# Generate titles for the code. I think the titles are a little fucked up, but let's look at that l8r.
	title_prefix = ""

	# Add "Technology" if sorting by technology
	if sorttype == 0:
		title_prefix += "Technology" + ": "  # Add colon after technology

	# Add "Cohort" or "Cohort and" based on number of firms and charts
	if FSnum_j > 1:
		title_prefix += "Cohort"
	else:
		if chrtnum_j > 1:
			title_prefix += "Cohort and "

	# Add "by" if there's any sorting or cohort information
	if title_prefix:
		title_prefix += "by "

	titlstr1 = title_prefix + titlstr2

	# Assuming typestr, FSnum_j, typestr3, grphtype are already defined

	# Combine title elements
	titlstr1 = typestr + titlstr1  # Add variable name to title

	# Format string for integer with leading zeros
	format_string = "%0*d"  # Use 'd' for integers (adjust width as needed)

	# Construct farm size string with leading zeros
	FSstr = "_" + format_string % (1, FSnum_j) + typestr3 + "_"  # Adjust width for FSnum_j

	# Set y-axis label based on graph type
	if grphtype > 10:
		yalabel = typestr  # Use variable name for y-axis
	else:
		yalabel = typestr + " (000s of 2011 dollars)"  # Add units to y-axis label

	_="""
	## This is the territory where we likely set up a bunch of matplotlib stuff ##
	Line-thickness, Linecolors, linetypes, etc.
	"""


	if quants_j == 0:  
		qnum_j = quants_j.shape[0]  # Get number of rows (quantiles)
		iQunt = 1  # Flag set to indicate presence of quantiles
	else:
		qnum_j = 0  # Set number of quantiles to 0
		iQunt = 0  # Flag set to indicate absence of quantiles

	for i in range(qnum_j):  # Use range(qnum_j) for guaranteed number of iterations
		iQunt = i + 1  # Adjust iQunt for zero-based indexing

		# Set quantile string based on value
		if iQunt == 0:
			quntstr = "Mean "
			quntstr2 = "avg"
		elif quants_j[i] == 0.5:
			quntstr = "Median "
			quntstr2 = f"{50:.2f}"  # Format 50 as string with 2 decimal places
		else:
			quntstr2 = f"{100*quants_j[i]:.2f}"  # Format 100*quantile as string with 2 decimal places
			quntstr = f"{quntstr2}th %tile "

			"""
			This is where they load in files from generated by grphmtx, pad them, and call doplot
			"""

			# Construct file name
			#fnamestr = makename(grphpath, typestr2 + "dt" + typestr3, iQunt)

			# Load data (assuming ^ is a data loading function)
			#alldat = load_alldat(fnamestr)  # Replace ^ with your actual data loading function

			# Get number of columns and rows
			#cn = alldat.shape[1]  # Use shape for columns
			#rn = alldat.shape[0]  # Use shape for rows

			# Add extra rows for better plotting (assuming ~miss creates missing value mask)
			#xtrarow1 = (alldat[0, 0] - 1) * ~np.isnan(np.ones((1, cn - 1)), axis=1)
			#xtrarow2 = (alldat[rn - 1, 0] + 1) * ~np.isnan(np.ones((1, cn - 1)), axis=1)
			#alldat = np.concatenate((xtrarow1[np.newaxis, :], alldat, xtrarow2[np.newaxis, :]), axis=0)

			# Call doplot

			#doplot(alldat, figtitle, figname, figdim)

			# Skip to next iteration if grphtype is 11 (Leased/Total Ratio)
			if grphtype == 11:
				iQunt += 1
				continue

			# Assuming alldat, figtitle, figname, figdim, grphpath, typestr2, typestr3, iQunt, cn, xtrarow1, xtrarow2 are defined
			_="""
			# Load simulated data (assuming ^ is a data loading function)
			fnamestr = makename(grphpath, typestr2 + "sm" + typestr3, iQunt)
			allsim = load_allsim(fnamestr)  # Replace ^ with your actual data loading function

			# Select relevant columns and add extra rows
			allsim = allsim[:, :cn]  # Select columns 1 to cn (inclusive)
			allsim = np.concatenate((xtrarow1[np.newaxis, :], allsim, xtrarow2[np.newaxis, :]), axis=0)

			# Plot simulated data
			doplot(allsim, figtitle, figname, figdim)

			# Create combined data (assuming alldat and allsim have same number of rows)
			allboth = np.logical_xor(alldat[:, 1:cn], allsim[:, 1:])  # Use np.logical_xor for XOR

			# Plot combined data
			doplot(allboth, figtitle, figname, figdim)
			"""


def doplot(alldat, figtitle, figname, figdim):
	print(f"plot: {figtitle}!")


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
