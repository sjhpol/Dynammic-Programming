import numpy as np

from variables import *
from functions import logitrv

from all_funcs import loaddat, initdist, datasetup, getgrids, makepvecs, onerun_c

# Get the grid points
sizevec, numvec, Astate, Estate, TAstate, Bstate, Kstate, lagKstate, NKState, Cstate = getgrids(Arange, Afine, grdcoef_lv, grdcoef_rt, Enum0, TAnum0, nodebt, Knum, lagKNum, Bnum0, NKnum0, Cnum0, K_min, 1)

# Get the data
IDs, owncap, obsmat, lsdcap, totcap, LTKratio, totasst, totliab, equity, cash, CAratio, debtasst, intgoods, nkratio, rloutput, ykratio, cashflow, dividends, divgrowth, dvgobsmat, DVKratio, DVFEratio, eqinject, goteqinj, grossinv, gikratio, netinv, nikratio, iobsmat, age_2001, init_yr, init_age, av_cows, famsize, cohorts, chrttype, milktype, herdtype, farmtype, avgage, datawgts, initstate = loaddat(timespan, np.array([1, 0]),1, chrtnum,chrtstep,sizescreen,wgtddata)

# Get the initial distribution of the variables
randrows, idioshks, numsims = initdist(IDs, farmtype, initstate, obsmat, iobsmat, dvgobsmat, np.array([bornage, retage]), cohorts, numsims, timespan+2, init_age, datawgts)

# Create dummy weights matrices
dumdwgts = np.ones_like(datawgts)
dumswgts = np.ones((numsims, timespan))

# Create Age vector
agevec = np.concatenate([agevec, np.array([numsims])])

# Save variables to path
np.savetxt(f'{iopath}agevec.txt', agevec)
np.savetxt(f'{iopath}sizevec.txt', sizevec)
np.savetxt(f'{iopath}gkvec.txt', gkvec)
np.savetxt(f'{iopath}Astate.txt', Astate)
np.savetxt(f'{iopath}Estate.txt', Estate)
np.savetxt(f'{iopath}TAstate.txt', TAstate)
np.savetxt(f'{iopath}Bstate.txt', Bstate)
np.savetxt(f'{iopath}Kstate.txt', Kstate)
np.savetxt(f'{iopath}lagKstate.txt', lagKstate)
np.savetxt(f'{iopath}NKState.txt', NKState)
np.savetxt(f'{iopath}Cstate.txt', Cstate)

	# Order = (1) baseline; (2) chi = 0; (3) psi = 0; (4) nu = 0; (5) nu = 0.25; (6) lam = 0.175; (7) chi=0 + lam=0.175; (8) w = 30;
# (9) no renegotiation; (10) herd-size weighted; (11) herd-size weighted, mean capital target

# Baseline parameters
bta = logitrv(np.array([0.972874])/betamax)
nu = np.log([4.341137])
# Coefficient of RRA
c_0 = np.log([3.665841])
# Flow utility curvature shifter
c_bar = np.log([12.51700])
# consumption threshold for bequest motive
finalMPC = logitrv(np.array([0.009502]))
# Minimum consumption
cfloor = np.log(1e-5)
# Flow utility from being a farmer, as consumption increment
chi0 = logitrv(np.array([0.337277]))
# Coefficient on owner labor
alp = logitrv(np.array([0.126302]))
alp = np.vstack((alp, logitrv(np.array([0.112983]))))
# Coefficient on capital
gam0 = logitrv(np.array([0.202019]))
gam0 = np.vstack((gam0, logitrv(np.array([0.136587]))))
# shift term in production function
nshft0 = np.log(1e-5)
# liquidation costs
lam = logitrv(np.array([0.35]))
# Capital downsizing costs
phi = logitrv(np.array([0.2]))
# liquidity constraint scaler
zta = np.log([2.640576])
# fixed operating cost
fcost0 = np.log(1e-10)
# colcnst*B <= K
colcnst = np.log([1.030748])
# Annual earnings as non-farmer, in thousands
w_0 = [15]
# 1 => no borrowing constraint
nocolcnst = [0]
# 0 => renegotiation if in default
noReneg = [0]

if numFTypes == 1:
	alp[1, :] = alp[0, :]
	gam0[1, :] = gam0[0, :]

# Assign values from parmvec and settvec
parmvec = np.vstack([bta, nu, c_0, c_bar, finalMPC, chi0, cfloor, alp, gam0, nshft0, lam, phi, zta, fcost0, colcnst])
settvec = np.vstack([w_0, nocolcnst, noReneg])
specnum = 0
parmvec = parmvec[:, specnum]
settvec = settvec[:, specnum]

w_0 = settvec[0]
wprof = w_0*np.ones(lifespan)
nocolcnst = settvec[1]
noReneg = settvec[2]

# Initialize other variables
nofcost = 1  # 1 => no fixed operating cost
inadaU = 0  # 1 => lim c->0 MUC = +oo
linprefs = (specnum == 3)  # 0 => nu = 0
nobeq = 0 + linprefs  # 1 => bequest motive shut down
nonshft = 1  # 1=> Standard Cobb-Douglas
noDScost = 1  # 1 => no penalty to selling capital
fixlam = 1  # 1=> do not estimate liquidation penalty

finparms0 = np.array([r_rf, bigR, bigG, gkE, noReneg, numFTypes])

np.savetxt(f'{iopath}wprof.txt', wprof)

# Make preference and financial parameter vectors
prefparms, finparms, gam, ag2, nshft, fcost = makepvecs(parmvec, betamax, linprefs, nobeq, w_0, bigR, numFTypes, inadaU, nonshft, noDScost, nofcost, nocolcnst, prnres, noReneg, finparms0)


TFPaggshks, TFP_FE, TFPaggeffs, tkqntdat, DAqntdat, CAqntdat, nkqntdat, gikqntdat, \
	ykqntdat, divqntdat, dvgqntdat, obsavgdat, tkqcnts, divqcnts, dvgqcnts, std_zi,\
	zvec, fevec, k_0, optNK, optKdat, countadj, real_data_matrixes = datasetup(gam, ag2, nshft, fcost, rloutput,
														   totcap, intgoods, obsmat, farmtype, av_cows, famsize,
														   datawgts, chrttype, iobsmat, dvgobsmat,
			  												dividends, divgrowth, LTKratio, debtasst, nkratio, gikratio,
			    											CAratio, ykratio, dumdwgts, avgage)
beta_nu = False
chi_lambda = True

if beta_nu:
	# Make grid
	Nbeta = 10
	Nnu = 10
	beta_list = np.linspace(0.85,0.99,Nbeta)
	nu_list = np.linspace(2, 5, Nnu)

	# Allocate
	obj = np.nan + np.zeros((Nbeta, Nnu))

	# Find objective function for each combination of beta and rho
	for idx, i in enumerate(beta_list):
		for jdx, j in enumerate(nu_list):
			print(f'(i, j): ({i, j})')
			# est_par = ['beta','nu']
			bta = logitrv(np.array([i])/betamax)
			nu = np.log(j)

			parmvec = np.vstack(
				[i, j, c_0, c_bar, finalMPC, chi0, cfloor, alp, gam0, nshft0, lam, phi, zta, fcost0, colcnst])
			parmvec = parmvec[:, specnum]

			prefparms, finparms, gam, ag2, nshft, fcost = makepvecs(parmvec, betamax, linprefs, nobeq, w_0, bigR, numFTypes,
																	inadaU, nonshft, noDScost, nofcost, nocolcnst, prnres,
																	noReneg, finparms0)

			numparms = parmvec.shape[0]
			fixvals = parmvec
			zerovec = 1
			obj[idx,jdx] = onerun_c(parmvec, fixvals, betamax, linprefs, nobeq, w_0, bigR, numFTypes, inadaU, nonshft, noDScost, nofcost,
				nocolcnst, prnres, noReneg, finparms0, idioshks, randrows,
			   rloutput, totcap, intgoods, obsmat,
			   farmtype, av_cows, famsize, datawgts, chrttype, iobsmat,
			   dvgobsmat, dividends, divgrowth, LTKratio, debtasst, nkratio,
			   gikratio, CAratio, ykratio, dumdwgts, numsims, avgage)       # TODO: replace with onerun that calls C code

if chi_lambda:

	Nchi = 10
	chi_range = np.linspace(0, 0.99, Nchi)

	Nlambda = 10
	lambda_range = np.linspace(0, 0.50, Nlambda)

	# Allocate
	obj = np.nan + np.zeros((Nchi, Nlambda))

	# Find objective function for each combination of beta and rho
	for idx, i in enumerate(chi_range):
		for jdx, j in enumerate(lambda_range):
			print(f'(i, j): ({i, j})')

			chi0 = logitrv(np.array([i]))
			lam = logitrv(np.array([j]))

			parmvec = np.vstack(
				[i, j, c_0, c_bar, finalMPC, chi0, cfloor, alp, gam0, nshft0, lam, phi, zta, fcost0, colcnst])
			parmvec = parmvec[:, specnum]

			prefparms, finparms, gam, ag2, nshft, fcost = makepvecs(parmvec, betamax, linprefs, nobeq, w_0, bigR,
																	numFTypes,
																	inadaU, nonshft, noDScost, nofcost, nocolcnst,
																	prnres,
																	noReneg, finparms0)

			numparms = parmvec.shape[0]
			fixvals = parmvec
			zerovec = 1
			obj[idx, jdx] = onerun_c(parmvec, fixvals, betamax, linprefs, nobeq, w_0, bigR, numFTypes, inadaU, nonshft,
								 noDScost, nofcost,
								 nocolcnst, prnres, noReneg, finparms0, idioshks, randrows,
								 rloutput, totcap, intgoods, obsmat,
								 farmtype, av_cows, famsize, datawgts, chrttype, iobsmat,
								 dvgobsmat, dividends, divgrowth, LTKratio, debtasst, nkratio,
								 gikratio, CAratio, ykratio, dumdwgts, numsims,
								 avgage)  # TODO: replace with onerun that calls C code

