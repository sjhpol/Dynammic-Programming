import numpy as np

# Define other variables
job = 1
basecase = 1
firstyr = 2001
lastyr = 2011
timespan = lastyr - firstyr + 1
yrseq = np.arange(0, timespan)
bornage = 20
retage = 75
lifespan = retage + 1 - bornage
ageseq = np.arange(bornage, lifespan + 1)
agevec = np.array([bornage, retage, lifespan, firstyr, lastyr, timespan])

numfarms = 363
mvcode = -99
statacrap = 0

wgtddata = 0  # 1 => Herd size weights
wgtdsplit = 0  # 1 => Herd size splits use weights
wgtdmmts = 0  # 1 => momwgts adjusted using herd size wgts
simtype = 1  # 1 => all ages; 2 => specific age
clonesims = 1  # 0 => bootstrap initial distribution; 1 => scale up
sepeqinjt = 0  # 0 => treat equity injections as neg dividends; 1 => separate
divbasemin = 1  # minimum abs value of denominator in div growth calculation
divlag = 1  # 1 => Match model t+1 dividend to data at time t
cashlag = 0  # 1 => Match model t+1 cash to data at time t
tfplag = 1
numFTypes = 2  # >1 => heterogeneous technology
sizescreen = [0, 1e6]  # 33.5 or 1147 depending on condition
sizevar = 2  # 1 => sort by TFP fixed effect; 2 => sort by # of cows
techsort = 1  # 0 => sort farm tech by herd size, 1 => sort by tech type
GMMsort = 0  # 0 => moments split by sizevar, 1 => split by techsort
cellmin = 2  # minimum number of observations needed to be in mmt vector
divmmts = 2  # 1 => include dividends in SMD criterion; 2 => include dividend growth
chrtnum = 2
chrtstep = 1 / chrtnum
FSstate = 0.5
quants_lv = 0.5
quants_rt = 0.5
quantmin = 10
checktie = 1
capscale = 1  # 1 => variables in charts are avgs of ratios; 2 => ratios of averages
prnres = 2 # print results
prngrph = 1 # print graphs

r_rf = 0.04  # Risk-free ROR
dlt = 0.055545  # depreciation rate
betamax = 1.05  # Upper limit on discount factor
bigR = 1 + r_rf  # 1 + rate of return
gkdim = 1  # dimension of capital price shock
std_gk = 0  # s.d. of capital gains shocks
# {gkpmtx, gkvals, intvals} = tauch(std_gk, 0, gkdim, .05, .05)
gkvals = 0.035582
gkpmtx = 1
gkE = gkpmtx * gkvals
gkvec = np.array([gkdim, 0, std_gk, gkvals, gkpmtx])
rdg = r_rf + dlt - gkvals
rdgE = r_rf + dlt - gkE
bigG = 1  # farm size growth

zdim = 8  # dimension of TFP shock
rho_z = 0.0  # autocorrelation
fedim = 6  # dimension of farm-specific mean productivity level (fixed effect)
cutoff_z = 0.0125  # Cut-off probs for Tauchen discretization
cutoff_fe = 0.025
farmbill = [0, 0, 1]  # 1st element = 1 => farm bill discretization;
# 2nd element = % cutoff; 3rd element = premium adjustment
aggshkscl = 1  # 0 => no aggregate TFP shocks
idioshkscl = 1  # 0 => no idiosyncratic transitory TFP shocks
momwgts = np.ones((8, 1))  # Scaling of moments and exit error penalty
momwgts[7] = 5
# Order: capital, dividends, debt/assets, cash/assets, i-goods/K, grInv/K, y/K, exit errs
parmtrans = 1  # Transform/scale paramzdimeters

grdcoef_lv = 0.55  # grid curvature parameters; lower value means concentrated near 0
grdcoef_rt = 1
Arange = 10000  # range of asset grid for worker
Afine = 2003 / 500  # fineness of asset grid
Gfine = 0.9
Enum0 = np.floor(78 * np.array(Gfine))  # E indicates net worth
TAnum0 = np.floor(65 * np.array(Gfine))  # TA indicates total assets
nodebt = 0  # 1=> cannot borrow
Knum = 101  # K indicates physical capital
K_min = 30  # Kmin >> 0 rules out subsistence farming
lagKNum = int((Knum - 1) / 5 + 1)  # Make capital adjustment grid coarse
Bnum0 = 101  # B indicates debt
NKnum0 = 16  # igoods/capital relative to theoretical best
Cnum0 = 45  # liquid assets

# Number of simulations
numsims = 90000

# Create Age vector
agevec = np.concatenate([agevec, np.array([numsims])])
