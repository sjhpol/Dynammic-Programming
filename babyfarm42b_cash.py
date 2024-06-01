import os
import pandas as pd
import json

from variables import *
from functions import logitrv

from all_funcs import loaddat, initdist, datasetup, getgrids, makepvecs, onerun, graphs, comp_stats

graph_time = False
comp_time = False

runnumber = ""

print(f"iopath: {iopath}")

outfile = rootdir + "babyfarm42b_cash.txt"
output_file = open(outfile, "w")
output_file.write("\n    ")  # Write date and time
output_file.close()

srchshks = np.random.randn(100, 50)
np.random.seed(12032013)
sttime = np.datetime64("today", "D")

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
bta = logitrv(np.array([0.972874, 0.999723, 1.000921, 0.967146, 0.972585, 0.972849, 1.000325, 0.972860, 0.980321, 1.0084573, 0.973077])/betamax)
nu = np.log([4.341137, 4.695300, 4.730706, 1.00E-10, 0.25, 4.339284, 4.441994, 4.342447, 4.143209, 10.2518451, 4.354765])
# Coefficient of RRA
c_0 = np.log([3.665841, 1.844800, 3.103255, 1.147466, 1.426730, 3.665839, 1.705993, 3.665333, 3.827312, 4.2142410, 3.673355])
# Flow utility curvature shifter
c_bar = np.log([12.51700, 20.35994, 14.96266, 21.83993, 28.14095, 12.51627, 20.77165, 12.52430, 12.823179, 13.7334848, 12.456519])
# consumption threshold for bequest motive
finalMPC = logitrv(np.array([0.009502, 0.0155645, 0.0155722, 0.999999, 0.0190932, 0.0094996, 0.016688, 0.009497, 0.008199, 0.0087726, 0.009549]))
# Minimum consumption
cfloor = np.log(1e-5) * np.ones(11)
# Flow utility from being a farmer, as consumption increment
chi0 = logitrv(np.array([0.337277, 1.00E-10, 0.221065, 0.836652, 0.4820314, 0.388852, 1E-10, 0.650041, 0.490080, 0.3428712, 0.337169]))
# Coefficient on owner labor
alp = logitrv(np.array([0.126302, 0.170062, 0.159017, 0.128515, 0.147956, 0.126254, 0.176291, 0.126278, 0.129890, 0.1163282, 0.126393]))
alp = np.vstack((alp, logitrv(np.array([0.112983, 0.211261, 0.178368, 0.106208, 0.116073, 0.112954, 0.222588, 0.112945, 0.114163, 0.1139515, 0.113131]))))
# Coefficient on capital
gam0 = logitrv(np.array([0.202019, 0.212099, 0.193745, 0.165296, 0.208204, 0.202115, 0.207468, 0.202029, 0.219456, 0.2065444, 0.201797]))
gam0 = np.vstack((gam0, logitrv(np.array([0.136587, 0.135600, 0.133331, 0.132061, 0.131892, 0.136496, 0.138325, 0.136546, 0.138566, 0.1384974, 0.136521]))))
# shift term in production function
nshft0 = np.log(1e-5) * np.ones(11)
# liquidation costs
lam = logitrv(np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.175, 0.175, 0.35, 0.35, 0.35, 0.35]))
# Capital downsizing costs
phi = logitrv(np.array([0.2])) * np.ones(11)
# liquidity constraint scaler
zta = np.log([2.640576, 2.434424, 2.790988, 2.599504, 2.509795, 2.641080, 2.437347, 2.640852, 2.502655, 3.2443405, 2.648191])
# fixed operating cost
fcost0 = np.log(1e-10) * np.ones(11)
# colcnst*B <= K
colcnst = np.log([1.030748, 0.960522, 1.00E-05, 1.083009, 1.095271, 1.030443, 0.986116, 1.030409, 1.060153, 0.8032255, 1.030991])
# Annual earnings as non-farmer, in thousands
w_0 = [15, 15, 15, 15, 15, 15, 15, 30, 15, 15, 15]
# 1 => no borrowing constraint
nocolcnst = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# 0 => renegotiation if in default
noReneg = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

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

# Clear variables
del bta, nu, c_0, cfloor, chi0, finalMPC, c_bar, alp, gam0, nshft0, lam, phi, zta, fcost0, colcnst

# Make preference and financial parameter vectors
prefparms, finparms, gam, ag2, nshft, fcost = makepvecs(parmvec, betamax, linprefs, nobeq, w_0, bigR, numFTypes, inadaU, nonshft, noDScost, nofcost, nocolcnst, prnres, noReneg, finparms0)


TFPaggshks, TFP_FE, TFPaggeffs, tkqntdat, DAqntdat, CAqntdat, nkqntdat, gikqntdat, \
	ykqntdat, divqntdat, dvgqntdat, obsavgdat, tkqcnts, divqcnts, dvgqcnts, std_zi,\
	zvec, fevec, k_0, optNK, optKdat, countadj, real_data_matrixes = datasetup(gam, ag2, nshft, fcost, rloutput,
														   totcap, intgoods, obsmat, farmtype, av_cows, famsize,
														   datawgts, chrttype, iobsmat, dvgobsmat, 
			  												dividends, divgrowth, LTKratio, debtasst, nkratio, gikratio,
			    											CAratio, ykratio, dumdwgts, avgage)

# datadscr her. Den laver graphs! (og sorterer med TFP, men det kan vi tage senere).
# (data describe)

# getaggrph
# get aggregate graphs her! Laver også graphs

numparms = parmvec.shape[0]
fixvals = parmvec
zerovec = 1 					# Note, that this becomes a vector later <_<. Also never has zeroes
onerun(parmvec, fixvals, betamax, linprefs, nobeq, w_0, bigR, numFTypes, inadaU, nonshft, noDScost, nofcost,
		  	nocolcnst, prnres, noReneg, finparms0, idioshks, randrows,
		   rloutput, totcap, intgoods, obsmat,
		   farmtype, av_cows, famsize, datawgts, chrttype, iobsmat,
		   dvgobsmat, dividends, divgrowth, LTKratio, debtasst, nkratio,
		   gikratio, CAratio, ykratio, dumdwgts, numsims, avgage)

if graph_time:

	graphs(gam, ag2, nshft, fcost, rloutput, totcap, intgoods, obsmat,
			   farmtype, av_cows, famsize, datawgts, chrttype, iobsmat,
			   dvgobsmat, dividends, divgrowth, LTKratio, debtasst, nkratio,
			   gikratio, CAratio, ykratio, dumdwgts, avgage, k_0, optNK, TFP_FE,
			   randrows, dumswgts, numsims)

if comp_time:
	beq = False
	l_ = False
	chi_ = False

	# Change in bequest tax => change in babyfarm18b.py
	if beq:
		comp_stats('beq', fcost, gam, ag2, nshft, k_0, optNK, TFP_FE, randrows)

	# Change in lambda: change above in lam variable
	if l_:
		comp_stats('lambda', fcost, gam, ag2, nshft, k_0, optNK, TFP_FE, randrows)

	# Change in chi: change above in chi variable
	if chi_:
		comp_stats('chi', fcost, gam, ag2, nshft, k_0, optNK, TFP_FE, randrows)

	# Change in lambda: change above in lam variable
	if l_ and chi_:
		comp_stats('chi_lambda', fcost, gam, ag2, nshft, k_0, optNK, TFP_FE, randrows)

	def load_comp():

		order = ['std', 'lambda', 'chi', 'chi_lambda', 'beq']

		total_df = pd.DataFrame()

		for file in os.listdir('comp_stats'):
			name, ext = os.path.splitext(file)

			f = open(f'comp_stats/{file}')
			data = json.load(f)
			del f
			temp_df = pd.DataFrame(data, index=[name])

			total_df = pd.concat([total_df, temp_df])

		std_df = total_df.loc['std', :]
		std_alive_frac = std_df['frac alive']

		total_df['frac alive'] = total_df['frac alive'] / std_alive_frac
		total_df = total_df.sort_index(key=lambda x: order)