# Other imports
from variables import *
from typing import TypeAlias, List, Tuple, Final # Try to statically type everything so we know wtf is going on.
import numpy as np # brug numpy så meget som muligt.
import math
import numpy.typing as npt
import os

# Speed!!!
import numba as nb
from numba import jit
from numba import njit
from numba import prange

Vector = npt.ArrayLike

# Parameter values
agevec       = np.loadtxt(f"{iopath}agevec.txt", dtype=float)
sizevec      = np.loadtxt(f"{iopath}sizevec.txt", dtype=float)
prefparmsvec = np.loadtxt(f"{iopath}prefparms.txt", dtype=float)
finvec     = np.loadtxt(f"{iopath}finparms.txt", dtype=float)

# the data we interpolate over
assetvec    = np.loadtxt(f"{iopath}assetvec.txt", dtype=float)
equityvec    = np.loadtxt(f"{iopath}equityvec.txt", dtype=float)
totassetvec = np.loadtxt(f"{iopath}totassetvec.txt", dtype=float)
debtvec     = np.loadtxt(f"{iopath}debtvec.txt", dtype=float)
capitalvec  = np.loadtxt(f"{iopath}capitalvec.txt", dtype=float)
lagcapvec   = np.loadtxt(f"{iopath}lagcapvec.txt", dtype=float) # Den er tom?
NKratiovec  = np.loadtxt(f"{iopath}NKratiovec.txt", dtype=float)
cashvec     = np.loadtxt(f"{iopath}cashvec.txt", dtype=float)

# Shocks
zvec            = np.loadtxt(f"{iopath}zvec.txt", dtype=float)
fevec           = np.loadtxt(f"{iopath}fevec.txt", dtype=float)
gkvec           = np.loadtxt(f"{iopath}gkvec.txt", dtype=float)
wageprofilevec  = np.loadtxt(f"{iopath}wageprofilevec.txt", dtype=float)

# Getting the shapes that we need for the interpolation vectors
assetNum: int    = assetvec.shape[0]
equityNum: int   = equityvec.shape[0]
totassetNum: int = totassetvec.shape[0]
debtNum: int     = debtvec.shape[0]
capitalNum: int  = capitalvec.shape[0]
lagcapNum: int   = lagcapvec.shape[0]
NKratioNum: int  = NKratiovec.shape[0]
cashNum: int     = cashvec.shape[0]

## PARAMETRE FRA MAKROER
# 'Final' her betyder, at vi ikke kan ændre dem! :)
printOn: Final[int] = 2        # 'bool' for whether to print or not.  (will probably depend on job (whether estimating or not), MPI, etc.
errorPrintOn: Final[int] = 1   # want separate tests for printing errors and printing other things.
rounder: Final[float] = 0.0001
ADDRESS_LEN: Final[int] = 300
HSDIM: Final[int] = 3          # Placeholder
tauBeq: Final[float] = 0.0       # 0.15 Marginal tax rate on bequests per De Nardi
exBeq: Final[int] = 600        # Bequest exemption level per De Nardi
minStepPct: Final[float] = 0.05  # Minimum search interval in one direction, as fraction of control space
gridShrinkRate: Final[float] = 0.03 # Rate at which interval shrinks, as fraction of control space
eRepayMin: Final[int] = 1      # Limits excessive borrowing in final period of life
taxDim: Final[int] = 7         # dim. of vector of marginal tax rates
eqInject: Final[int] = 1       # allow dividends to be negative up to c_0

taxBrk: Final[npt.ArrayLike] = np.array([7490, 48170, 89150, 112570, 177630, 342120])
taxMar: Final[npt.ArrayLike] = np.array([0.0765, 0.2616, 0.4119, 0.3499, 0.3834, 0.4360, 0.4761])

#    assert len(taxBrk) == taxDim - 1, f"Tax brackets list must contain exactly {taxDim - 1} elements."
#    assert len(taxMar) == taxDim, f"Tax marginal rates list must contain exactly {taxDim} elements."
incomeBrk = np.zeros(taxDim - 1)  # Initialize with zeros

###### globderef
rounder: float = 0.0001

bornage: int  = int(math.floor(rounder + agevec[0]))
retage: int   = int(math.floor(rounder + agevec[1]))
lifespan: int = int(math.floor(rounder + agevec[2]))
firstyear: int= int(math.floor(rounder + agevec[3]))
lastyear: int = int(math.floor(rounder + agevec[4]))
timespan: int = int(math.floor(rounder + agevec[5]))
numSims: int  = int(math.floor(rounder + agevec[6]))

assetMin: float   = sizevec[0]
assetMax: float   = sizevec[1]
equityMin: float  = sizevec[2]
equityMax: float  = sizevec[3]
totassetMin: float= sizevec[4]
totassetMax: float= sizevec[5]
debtMin: float    = sizevec[6]
debtMax: float    = sizevec[7]
capitalMin: float = sizevec[8]
capitalMax: float = sizevec[9]
cashMin: float    = sizevec[10]
cashMax: float    = sizevec[11]

beta: float     = prefparmsvec[0]
nu: float       = prefparmsvec[1]
nu2: float      = prefparmsvec[2]
c_0: float      = prefparmsvec[3]
c_1: float      = prefparmsvec[4]
thetaB: float   = prefparmsvec[5]
chi: float      = prefparmsvec[6]  # Jones: Here it is a consumption increment
consFloor: float= prefparmsvec[7]

cscale: float = pow(c_0 + wageprofilevec[0], -nu2) # ?
chi = cscale * (1 / nu2) * (pow(c_0 + wageprofilevec[0], nu2) - pow(c_0 + wageprofilevec[0] - chi, nu2)) # ?

alpha1: float         = finvec[0]
alpha2: float         = finvec[1]
gamma1: float         = finvec[2]
gamma2: float         = finvec[3]
igshift: float        = finvec[4]
_lambda: float        = finvec[5]  # Using _lambda instead of lambda to avoid conflict with reserved keyword
phi: float            = finvec[6]
zeta: float           = finvec[7]
delta: float          = finvec[8]
fixedcost: float      = finvec[9]
psi_inverse: float    = finvec[10]  # fraction of capital that can be borrowed, inverted. Negative => no limit
r_riskfree: float     = finvec[11]
bigR: float           = finvec[12]
bigG: float           = finvec[13]
eGK: float            = finvec[14]  # Expected value
noRenegotiation: int  = int(round(rounder + finvec[15]))
ftNum: int            = int(round(rounder + finvec[16]))

eRDG: float           = r_riskfree + delta - eGK
SLfrac: float         = phi * (1 - delta + eGK) / bigG  # irreversibility costs for t-1 capital

"""#UNDERRUTINER
###(skal før hovedoutput + husk type-hints)

"""

# This function defines production parameters. We have two different different pairs of values (alpha1, gamma1); (alpha2, gamma2)
# and associated exponents that we're interested in using. This returns the whole set.
@jit(nopython=True)
def prodFnParms(alpha0: float, gamma0: float) -> tuple[float, float, float, float, float]:
  alpha: float = alpha0
  gamma: float = gamma0
  ag2: float   = 1 - alpha - gamma
  ag3: float   = 1 / (alpha + gamma)
  gag: float   = (gamma)*(ag3)
  agag: float  = pow(ag2, ag3)

  return alpha, gamma, ag2, ag3, gag, agag

# alpha, gamma, ag2, ag3, gag, agag = prodFnParms(alpha1, gamma1)

# alpha, gamma, ag2, ag3, gag, agag = prodFnParms(alpha2, gamma2)

# double getUtility(double cons);
# This is just our utility function. cons = consumption. easy!
@jit(nopython=True, fastmath=True)
def getUtility(cons: float) -> float:
    if nu < 0:
        print("getUtility(): passed nu==0, exiting.")
        exit(1)

    utility = -999
    consAdj = c_0 + cons
    if nu == 1.0:
        utility = cscale * np.log(consAdj)
    elif nu == 0.0:
        utility = cscale * cons
    else:
        utility = cscale * pow(consAdj, 1-nu) / (1-nu)

    return utility

print(getUtility(5.0))

# Calculates how much bequests you make. Based on Denardi.
# tauBeq is the arveskatterate (tau bequest).
# exBeq (exemption bequest) is the arveskattefradrag (don't tax the first e.g. 600 kr.).
@jit(nopython=True)
def NetBequest(amountBequestedP: float) -> float:
    # Hvis arven overstiger fradraget
    if amountBequestedP > exBeq:
        net_bequest = exBeq + (1 - tauBeq) * (amountBequestedP - exBeq)
    else:
        net_bequest = amountBequestedP

    # Check that its not too low
    if net_bequest < consFloor:
        net_bequest = consFloor
    return net_bequest


print(NetBequest(1000.0)) # Burde ikke ændre noget, da skatterate = 0%

# utility from a net bequest for a single person
# phi_j(b_net) = phi_j*( (b_net+K_j)^(1-nu) )/(1-nu)
# not discounted
@jit(nopython=True)
def UBeqSingle(netBequestP: float) -> float:
    if nu == 1:
      utils  =  cscale * thetaB * np.log(netBequestP / (thetaB+1e-20) + c_1)
    else:
      utils  =  cscale * thetaB * pow(netBequestP / (thetaB+1e-20) + c_1, nu2) / nu2

    return utils

UBeqSingle(100.0)

# double AfterTaxIncome(double y);
# Returns post-tax income. Run it often. Very good!
@jit(nopython=True)
def AfterTaxIncome(y: float) -> float:
    if y < 0:
        # This case will be ruled out in maximization
        return -1
    elif y < taxBrk[0]:
        return (1 - taxMar[0]) * y
    elif y < taxBrk[1]:
        return incomeBrk[0] + (1 - taxMar[1]) * (y - taxBrk[0])
    elif y < taxBrk[2]:
        return incomeBrk[1] + (1 - taxMar[2]) * (y - taxBrk[1])
    elif y < taxBrk[3]:
        return incomeBrk[2] + (1 - taxMar[3]) * (y - taxBrk[2])
    elif y < taxBrk[4]:
        return incomeBrk[3] + (1 - taxMar[4]) * (y - taxBrk[3])
    elif y < taxBrk[5]:
        return incomeBrk[4] + (1 - taxMar[5]) * (y - taxBrk[4])
    else:
        return incomeBrk[5] + (1 - taxMar[6]) * (y - taxBrk[5])

print(AfterTaxIncome(25000))
print(AfterTaxIncome(-10))

# You pass in 3 list of floats, and it sets the third list.
# after-tax income at bracket points, used to calculate the vector, that we need for post-tax calculations
# only needs to be run once
# Could be refactored to exist without sideeffects
@jit(nopython=True)
def IncomeAtBrk(taxBrk: Vector, taxMar: Vector, incomeBrk: Vector) -> Vector:
    incomeBrk[0] = (1 - taxMar[0]) * taxBrk[0]  # The leftmost interval

    for j in range(1, len(taxBrk)):
        incomeBrk[j] = incomeBrk[j - 1] + (1 - taxMar[j]) * (taxBrk[j] - taxBrk[j - 1])

# double getBaseRevenues(double capital, double igoods, double gamma, double ag2);
# Does the prod. fnct. (4), without z_{qt}
# I think ...
# igoods is (n_t) "intermediate goods", which has a shifter, apparently
# ag2 is (1-\alpha-\gamma)
# can also easily be refactored.
@jit(nopython=True)
def getBaseRevenues(capital: float, igoods: float, gamma: float, ag2: float) -> float:
    return pow(capital, gamma) * pow(igoods + igshift, ag2)


getBaseRevenues(100.0, 50.0, 2.0, 2.0)

# Locates nearest point _below_ x in a sorted array
# Gives the dimension. Breaks in edge-cases.
# int Locate(double *Xarray, double x, int DIM)
@jit(nopython=True)
def Locate(Xarray: npt.ArrayLike, x: float) -> int:
    idx = np.searchsorted(Xarray, x, side='left') - 1

    return max(idx, 0) # Bounds our index to the left

# Example usage:
Xarray = np.array([1, 3, 5])  # Assuming Xarray is sorted in ascending order

Xarray_unsorted = np.array([5,3,1])
x = 15  # Target value

nearest_below_index = Locate(Xarray, x)
print(f"Index of nearest point below x {x}:", nearest_below_index)

nearest_below_index_unsorted = Locate(Xarray_unsorted, x)
print(f"Index of (unsorted) nearest point below x {x}:", nearest_below_index)

# locateClosest is just an inferior version of Locate. Wunderbar.

# The struct is just an index and an interpolation weight. But do we have (interpolation weights?).
# Bertel hasn't mentioned them at all. Let's look past this for a second.

# This function is used in the big loops a lot. Very important.
# The weights tell you how close you are. Makes it speedy, by lowering the total amount of needed grid points!
@jit(nopython=True, fastmath=True)
def GetLocation(xP: np.ndarray, x: float) -> Tuple[int, float]:
    j = Locate(xP, x) #+ 1
    Ind1 = j - 1

    # Check if the denominator is not zero before performing division
    if xP[j] - xP[j - 1] != 0:
        weight = (xP[j] - x) / (xP[j] - xP[j - 1])
    else:
        weight = 0.5  # Default value. Right down the middle. Needed to stop div. zero faults (+ to make the compiler stop whining)

    if weight > 1:
        weight = 1
    if weight < 0:
        weight = 0
    return Ind1, weight

# test
print(GetLocation(assetvec, 2.1))

# double getbaseIGoods(double capital, double eTFP, double gag, double agag, double ag3)
# Finding ideal igoods given capital stock and eTFP (e_total factor productivity. What is 'e'?).
# "Intermediate goods". What eqn. does this correspond to? Let's cool on this for a second.
# Idk but its w/e. Let's just plug in the logic.
@jit(nopython=True)
def getBaseIGoods(capital: float,
                  eTFP: float,
                  gag: float,
                  agag: float,
                  ag3: float,
                  igShift: float) -> float:
    igoods = agag * pow(capital, gag) * pow(eTFP, ag3) - igShift
    if igoods < 0:
        igoods = 0
    return igoods


# Used for the markov chains and shocks.
# creates a transition matrix
@jit(nopython=True)
def GetCDFmtx(nRows, nCols, transmtx):
    transCDFmtx = np.zeros((nRows, nCols + 1))

    for iRow in range(nRows):
        sum_val = 0
        for iCol in range(nCols):
            transCDFmtx[iRow][iCol] = sum_val
            sum_val += transmtx[iRow][iCol]
        # Adding a small value to the last element to avoid floating point precision issues
        transCDFmtx[iRow][nCols] = sum_val + 1e-10

    return transCDFmtx

# Takes in the very specially formatted GAUSS data
# where 0 is the shock, 1 is rho, 2 is std.dev., and 3-numStates is the progression
# Vi skal lige kigge mere på type hints når det er Numpy vi har at gøre med.
# used for the shocks section
@jit(nopython=True)
def getMarkovChain(chainInfoPtr: npt.ArrayLike, numStates: int) -> npt.ArrayLike:
    # Extracting parameters

    rho = chainInfoPtr[1]
    # std = chainInfoPtr[2]

    values = np.zeros(numStates)
    piInvarV = np.zeros(numStates)

    # Initializing transmtx
    transmtx = np.empty((numStates, numStates))

    for iState in range(numStates):
        values[iState] = chainInfoPtr[3 + iState]
        kState = iState if rho != 0 else 0
        transmtx[iState] = chainInfoPtr[3 + numStates * (kState + 1):3 + numStates * (kState + 2)]

    # Finding Invariant distribution
    ProductMat = transmtx.copy()
    subtot = np.zeros((numStates, numStates))

    hMax = 2000 if rho != 0 else 0

    for h in range(hMax):
        for iState in range(numStates):
            for jState in range(numStates):
                subtot[iState][jState] = np.sum(ProductMat[iState] * transmtx[:, jState])
                ProductMat[iState][jState] = subtot[iState][jState]

    piInvarV[:] = ProductMat[0]

    return transmtx, values, piInvarV

# Get cumulative distribution function vector. A vector that that turns a *invar* dist to cdf
# used for the markov chains & shocks.
@jit(nopython=True)
def GetCDFvec(nCols: int, probvec: npt.ArrayLike) -> npt.ArrayLike:
    sum_: float = 0.0
    CDFvec: npt.ArrayLike = np.zeros(nCols + 1)

    for iCol in range(nCols):
        CDFvec[iCol] = sum_
        sum_ += probvec[iCol]

    CDFvec[nCols] = sum_ + 1e-10

    return CDFvec

# Expected utility from leaving bequest matrix for a single, not discounted
# This one sets the utility variable, that we need for setting the the valFuncWork vector.
# We take in a empty bequestUM, set it along the states of the asset state points.
# Prob. the utility that you get from arv at each asset state.
# This can also be refactored to be less silly.
@jit(nopython=True)
def getUtilityBeq(bequestUM: npt.ArrayLike,
                  aState: npt.ArrayLike) -> npt.ArrayLike:
    for aInd in range(assetNum):
        bequestUM[aInd] = UBeqSingle(NetBequest(aState[aInd]))

    return bequestUM

def getAssignmentVec(numPoints, numNodes):
    numBlocks = math.ceil(numPoints / numNodes)
    assignmentVec = np.zeros((numPoints),dtype=int) # IDK about this dtype. Look more

    iPoint = 0
    for iNode in range(numNodes):
        for iBlock in range(numBlocks):
            thisPoint = iBlock * numNodes + iNode + 1
            if thisPoint > numPoints:
                continue
            assignmentVec[iPoint] = thisPoint - 1
            iPoint += 1

    return assignmentVec

@jit(nopython=True, fastmath=True)
def getExpectation(RandVar: np.ndarray,
                   NPTAIndex: np.ndarray,
                   NPTAWeight: np.ndarray,
                   iLagK: int,
                   iLagKwgt: float,
                   iDebt: int,
                   zProbs: np.ndarray) -> float:
    expectsum = 0.0

    if lagcapNum == 1:
        for iZNP in range(zNum):
            iNPTA = int(NPTAIndex[iZNP]) #mod
            iZNP2 = iZNP
            if zNum2 == 1:
              iZNP2 = 0
            #avg = NPTAWeight[iZNP] * RandVar[iZNP2, 0, iNPTA, iDebt] + (1 - NPTAWeight[iZNP]) * RandVar[iZNP2, 0, iNPTA + 1, iDebt] # ORIGINAL
            avg = NPTAWeight[iZNP] * RandVar[iZNP2, 0, iNPTA, iDebt] + (1 - NPTAWeight[iZNP]) * RandVar[iZNP2, 0, iNPTA, iDebt] # ORIGINAL
            expectsum += zProbs[iZNP] * avg
    else:
        for iZNP in range(zNum):
            iNPTA = int(NPTAIndex[iZNP]) # MOD
            iZNP2 = iZNP
            if zNum2 == 1:
              iZNP2 = 0

            #avg = NPTAWeight[iZNP] * RandVar[iZNP2, iLagK, iNPTA, iDebt] + (1 - NPTAWeight[iZNP]) * RandVar[iZNP2, iLagK, iNPTA + 1, iDebt] # ORIGINAL
            #avg2 = NPTAWeight[iZNP] * RandVar[iZNP2, iLagK + 1, iNPTA, iDebt] + (1 - NPTAWeight[iZNP]) * RandVar[iZNP2, iLagK + 1, iNPTA + 1, iDebt]
            avg = NPTAWeight[iZNP] * RandVar[iZNP2, iLagK, iNPTA, iDebt] + (1 - NPTAWeight[iZNP]) * RandVar[iZNP2, iLagK, iNPTA, iDebt]
            avg2 = NPTAWeight[iZNP] * RandVar[iZNP2, iLagK + 1, iNPTA, iDebt] + (1 - NPTAWeight[iZNP]) * RandVar[iZNP2, iLagK + 1, iNPTA, iDebt]
            avg = avg * iLagKwgt + avg2 * (1 - iLagKwgt)
            expectsum += zProbs[iZNP] * avg

    return expectsum

# Debugging code for the above.
_="""
            print("---")
            print(f"iZNP {iZNP}")
            print(f"iZNP2 {iZNP2}")
            print(f"iLagK {iLagK}")
            print(f"iNPTA {iNPTA}")
            print(f"iDebt {iDebt}")
            print(f'NPTAWeight.shape {NPTAWeight.shape}')
            print(f'RandVar.shape {RandVar.shape}')"""

"""# MAIN"""

# GetClosest lagged capital
@jit(nopython=True)
def getclosestLK(numPTrue: int,
                 truevec: np.ndarray,
                 targetvec: np.ndarray,
                 CLKindvec: np.ndarray,
                 CLKwgtvec: np.ndarray) -> None:

    for iTV in range(numPTrue):
        Ind1, weight = GetLocation(targetvec, truevec[iTV])
        CLKindvec[iTV] = Ind1
        CLKwgtvec[iTV] = weight

    return CLKindvec, CLKwgtvec

CLKindvec = np.zeros((capitalNum), dtype=int)
CLKwgtvec = np.zeros((capitalNum))

getclosestLK(capitalNum, capitalvec, lagcapvec, CLKindvec, CLKwgtvec)

### Shocks + Markov chains ###

# z (TFP)
rhoZ = zvec[1]
zNum = int(np.floor(rounder + zvec[0]))
zTransmtx, zValues, zInvarDist = getMarkovChain(zvec, zNum)
zNum2 = zNum  # dimension for state vector
if rhoZ == 0:
    zNum2 = 1
zTransCDFmtx = GetCDFmtx(zNum, zNum, zTransmtx)
zInvarCDF = GetCDFvec(zNum, zInvarDist)

# fixed effects (productivity!) (TFP)
feNum = int(np.floor(rounder + fevec[0]))
feProbs2, feValues, __ = getMarkovChain(fevec, feNum)

# gk = gains on capital. Conditionally i.i.d.
rhoGK = gkvec[1]
gkNum = int(np.floor(rounder + gkvec[0]))  # Capital gains shock, conditionally i.i.d.
gkTransmtx, gkValues, gkInvarDist = getMarkovChain(gkvec, gkNum)
gkNum2 = gkNum  # dimension for state vector
if rhoGK == 0:
    gkNum2 = 1
gkTransCDFmtx = GetCDFmtx(gkNum, gkNum, gkTransmtx)
gkInvarCDF = GetCDFvec(gkNum, gkInvarDist)

# done. TODO: Test lmao

# Initialize matrices
bequestUM: float = np.nan + np.zeros(assetNum)
bequestUM = getUtilityBeq(bequestUM,assetvec) #can be run after setting the py file up.
IncomeAtBrk(taxBrk, taxMar, incomeBrk)

recsize = (lifespan+1)*assetNum
#GetUtilityBeq(bequestUM,assetvec)      # Find utility from bequest matrix
IncomeAtBrk(taxBrk, taxMar, incomeBrk) # after-tax income at bracket points

recsize         = (lifespan+1)*assetNum
valfFuncWPtr    = np.nan + np.zeros(recsize)
recsize         = lifespan*assetNum
bestCWPtr       = np.nan + np.zeros(recsize) # optimal consumption choices
bestNPIWPtr     = np.nan + np.zeros(recsize) # index number of optimal time-t+1 assets

valfuncWork     = valfFuncWPtr.reshape(lifespan+1,assetNum)
valfuncWork[lifespan] = bequestUM
bestCWork       = bestCWPtr.reshape(lifespan,assetNum)
bestNPIWork     = bestNPIWPtr.reshape(lifespan,assetNum)

# Function to compute rules for a range of assets
@jit(nopython=True)
def GetRulesWorker(assetvec, wageProfile, valfuncWork, bestCWork, bestNPIWork):

    # Just for speed (Vroom!). Real value is 65.
    #lifespan = 5
    #

    for tInd in range(lifespan - 1, -1, -1):
        print(f"tInd {tInd}")
        for aInd, asset in enumerate(assetvec):
            # Compute cash-on-hand (GROSS OF TAXES), before any transfers
            cashonhand = bigR * assetvec[aInd] + wageProfile[tInd]
            NPIlim = 0

            for aNPInd in range(assetNum):
                diff = cashonhand - bigG * assetvec[aNPInd] - consFloor
                if diff < 0:
                    break

            NPIlim = aNPInd  # maximum feasible saving + 1
            if NPIlim < 1:
                NPIlim = 1

            # Initialize maximum to a value that can be exceeded for sure
            todaysUtility = getUtility(consFloor / 2 - c_0 / 2)
            continuationValue = valfuncWork[tInd + 1][0]
            value = todaysUtility + beta * continuationValue
            maxValue = value - 1e6
            oldMaxV = maxValue
            oldoldMaxV = oldMaxV
            maxCons = -1
            maxNPI = -1
            aNPImin = 0
            aNPImax = NPIlim

            if tInd < lifespan - 1:
                stepPct = minStepPct + pow((1 - gridShrinkRate), (lifespan - 1 - tInd))
                numStepsF = stepPct * assetNum
                numSteps = int(numStepsF)
                aNPImin = int(bestNPIWork[tInd + 1][aInd]) - numSteps
                aNPImax = aNPImin + 2 * numSteps + 1
                if aNPImin < 0:
                    aNPImin = 0
                if aNPImax > NPIlim:
                    aNPImax = NPIlim

            for aNPInd in range(aNPImin, aNPImax):
                # loop over decision variable, all possible savings levels tomorrow
                cons = cashonhand - bigG * assetvec[aNPInd]
                if cons < consFloor:
                    cons = consFloor
                todaysUtility = getUtility(cons)
                continuationValue = valfuncWork[tInd + 1][aNPInd]
                value = todaysUtility + beta * continuationValue

                if value > maxValue:
                    oldoldMaxV = oldMaxV
                    oldMaxV = maxValue
                    maxValue = value
                    maxCons = cons
                    maxNPI = aNPInd

                # If value function is decreasing in savings for two consecutive times we quit.
                # Makes sense if objective is concave.
                if value < oldMaxV and value < oldoldMaxV and aNPInd > 3:
                    break

            valfuncWork[tInd][aInd] = maxValue
            bestCWork[tInd][aInd] = maxCons
            bestNPIWork[tInd][aInd] = maxNPI

    return valfuncWork, bestCWork, bestNPIWork 


valfuncWork, bestCWork, bestNPIWork = GetRulesWorker(assetvec, wageprofilevec, valfuncWork, bestCWork, bestNPIWork)

# Calculate mapping from TotAssets x Lagged Capital x Debt to post-Liquidation Net Worth/Assets
# These are age-invariant, financial calculations

postliqAssets = np.zeros((lagcapNum, totassetNum, debtNum))
postliqNetWorth = np.zeros((lagcapNum, totassetNum, debtNum))
postliqNWIndex = np.zeros((lagcapNum, totassetNum, debtNum))
retNWIndex       = np.zeros((lagcapNum, totassetNum, debtNum));

@jit(nopython=True)
def getliqIndex(assetvec: np.ndarray, totassetvec: np.ndarray, lagcapvec: np.ndarray,
                debtvec: np.ndarray, postliqAssets: np.ndarray, postliqNetWorth: np.ndarray,
                postliqNWIndex: np.ndarray, SLfrac: float, lambda_: float) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:

    for iLagK in range(lagcapNum):
        sellingLoss = SLfrac * lagcapvec[iLagK]  # If there are capital irreversibilities
        for iTotAsset in range(totassetNum):
            thisAsset = (1 - lambda_) * (totassetvec[iTotAsset] - sellingLoss)
            thisAsset = max(0, thisAsset)  # Firm may experience massive operating loss

            for iDebt in range(debtNum):
                thisNetWorth = thisAsset - debtvec[iDebt]  # Debt is defined to be non-negative
                thisNetWorth = max(0, thisNetWorth)
                closestInd = Locate(assetvec, thisNetWorth)

                postliqAssets[iLagK, iTotAsset, iDebt] = thisAsset
                postliqNWIndex[iLagK, iTotAsset, iDebt] = closestInd
                postliqNetWorth[iLagK, iTotAsset, iDebt] = assetvec[closestInd]

    return postliqAssets, postliqNetWorth, postliqNWIndex

postliqAssets, postliqNetWorth, postliqNWIndex = getliqIndex(assetvec, totassetvec, lagcapvec, debtvec, postliqAssets, postliqNetWorth, postliqNWIndex, SLfrac, _lambda)

#retNWIndex = np.zeros((lagcapNum, totassetNum, debtNum)) # Special. Not defined by the previous call

# Calculate mapping from farmer's decision grid (total Capital x FE x NKratio x Cash expenditures ) to future (total) assets

_="""
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  GETNPTOTASSETS:  Go over farmer's decision grid, and figure out future total  */
/*                   assets associated with each decision.                        */
/*                   Since dividends are the residual to the budget constraint,   */
/*                   These calculations need only be done once.                   */
/*                   Note that capital adjustment costs are subtracted from the   */
/*                   that were issued PRIOR to these production sequences.        */"""
@jit(nopython=True)
def getNPtotAssets(capvec: np.ndarray,
                   capvecNum: int,
                   NKratiovec: np.ndarray,
                   cashvec: np.ndarray,
                   feValues: np.ndarray,
                   zValues: np.ndarray,
                   totassetvec: np.ndarray,
                   NPtotassetWeight: np.ndarray,
                   NPtotassetIndex: np.ndarray,
                   goodGridPoint: np.ndarray) -> None:

    for iFType in range(ftNum):
        print(f"iFType {iFType} / {ftNum}")

        if iFType == 0:
            alpha, gamma, ag2, ag3, gag, agag = prodFnParms(alpha1, gamma1)
        else:
            alpha, gamma, ag2, ag3, gag, agag = prodFnParms(alpha2, gamma2)

        for iTotK in range(capvecNum):
            totCapital = capvec[iTotK]
            for iFE in range(feNum):
                baseIGoods = getBaseIGoods(totCapital, feValues[iFE], gag, agag, ag3, igshift)
                for iNKrat in range(NKratioNum):
                    igoods = baseIGoods * NKratiovec[iNKrat]
                    baseRevenues = getBaseRevenues(totCapital, igoods, gamma, ag2)
                    expenses = igoods + fixedcost

                    for iCash in range(cashNum - 1, -1, -1):
                        thisCash = cashvec[iCash]
                        if thisCash < expenses / zeta:
                            break  # Check cash-in-advance constraint
                        goodGridPoint[iTotK, iFType, iFE, iNKrat, iCash] = 1

                        for iZNP in range(zNum):
                            totAssetsNP = ((1 - delta + eGK) * totCapital + feValues[iFE] * zValues[iZNP] * baseRevenues - expenses + thisCash) / bigG
                            point, weight = GetLocation(totassetvec, totAssetsNP)                    #, totassetNum)
                            NPtotassetIndex[iTotK, iFType, iFE, iNKrat, iCash, iZNP] = point      #.Ind1
                            NPtotassetWeight[iTotK, iFType, iFE, iNKrat, iCash, iZNP] = weight

    return NPtotassetWeight, NPtotassetIndex, goodGridPoint

NPtotassetWeight = np.zeros((capitalNum, ftNum, feNum, NKratioNum, cashNum, zNum))
NPtotassetIndex  = np.zeros((capitalNum, ftNum, feNum, NKratioNum, cashNum, zNum))
goodGridPoint    = np.zeros((capitalNum, ftNum, feNum, NKratioNum, cashNum))
NPtotassetWeight, NPtotassetIndex, goodGridPoint = getNPtotAssets(capitalvec, capitalNum, NKratiovec, cashvec, feValues, zValues,
                                                totassetvec, NPtotassetWeight, NPtotassetIndex, goodGridPoint)

# Final stretch. Wall of answers

liqDecisionMat = np.zeros((lifespan + 1, ftNum, feNum, zNum2, lagcapNum, totassetNum, debtNum))
valfuncMat = np.zeros((lifespan + 1, ftNum, feNum, zNum2, lagcapNum, totassetNum, debtNum))
fracRepaidMat = np.zeros((lifespan + 1, ftNum, feNum, zNum2, lagcapNum, totassetNum, debtNum))
valfuncFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum))
bestIntRateFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum))
bestKIndexFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum), dtype=int)
bestKFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum))
bestNKratIndexFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum), dtype=int)
bestNKratFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum))
bestCashIndexFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum), dtype=int)
bestCashFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum))
bestDividendFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum))
bestDebtIndexFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum), dtype=int)
bestDebtFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum))
bestNPTAWeightFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum, zNum))
bestNPTAIndexFarm = np.zeros((lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum, zNum), dtype=int)

@jit(nopython=False)
def getFinalLiq(totassetvec: np.ndarray,
                lagcapvec: np.ndarray,
                debtvec: np.ndarray,
                assetvec: np.ndarray,
                retNWIndex: np.ndarray,
                liqDecisionMat: np.ndarray,
                valfuncMat: np.ndarray,
                fracRepaidMat: np.ndarray) -> None:

    for iLagK in range(lagcapNum):
        if iLagK % 4 == 0:  # report less in the output buffer
          print(f"lagcap {iLagK} / {lagcapNum-1}")

        sellingLoss = SLfrac * lagcapvec[iLagK]  # If there are capital irreversibilities # H: what is 'capital irreversibitlies'?!
        for iTotAsset in range(len(totassetvec)):
            thisAsset = totassetvec[iTotAsset] - sellingLoss  # No liquidation costs for retirement
            if thisAsset < 0:
                thisAsset = 0  # Firm may experience massive operating loss

            for iDebt in range(debtNum):
                thisDebt = debtvec[iDebt]
                newDebt = thisDebt
                if newDebt > thisAsset:
                    newDebt = thisAsset
                finalAsset = thisAsset - newDebt
                retNWIndex[iLagK, iTotAsset, iDebt] = Locate(assetvec, finalAsset)
                thisUtil = UBeqSingle(NetBequest(finalAsset))
                thisFrac = 1
                if thisDebt > 0:
                    thisFrac = newDebt / thisDebt
                for iFType in range(ftNum):
                    for iFE in range(feNum):
                        for iZ in range(zNum2):
                            valfuncMat[lifespan, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = thisUtil
                            liqDecisionMat[lifespan, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = 1
                            fracRepaidMat[lifespan, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = thisFrac

    return retNWIndex, liqDecisionMat, valfuncMat, fracRepaidMat



# retained net worth index?
retNWIndex, liqDecisionMat, valfuncMat, fracRepaidMat = getFinalLiq(totassetvec, lagcapvec, debtvec, assetvec,
                                                          retNWIndex, liqDecisionMat, valfuncMat, fracRepaidMat);

# This is going to be a - to debug.
@jit(nopython=True, fastmath=True, parallel=True)
def getliqDecision(totassetvec: np.ndarray,
                   debtvec: np.ndarray,
                   equityvec: np.ndarray,
                   postliqAssets: np.ndarray,
                   postliqNWIndex: np.ndarray,
                   valfuncWork: np.ndarray,
                   valfuncFarm: np.ndarray,
                   liqDecisionMat: np.ndarray,
                   valfuncMat: np.ndarray,
                   fracRepaidMat: np.ndarray,
                   ageInd: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    for iLagK in range(lagcapNum):
        #print(f"iLagK {iLagK} / {lagcapNum}")
        for iTotAsset in prange(len(totassetvec)):
            thisAsset = totassetvec[iTotAsset]
            for iDebt in range(debtNum):
                thisDebt = debtvec[iDebt]

                iLiq = int(postliqNWIndex[iLagK, iTotAsset, iDebt])
                liqValue = valfuncWork[ageInd, iLiq]
                if noRenegotiation == 1:
                    thisEquity = thisAsset - thisDebt
                    if thisEquity < 0:
                        liqDec = 1
                        thisVal = liqValue
                        thisFrac = 1
                        newDebt = postliqAssets[iLagK, iTotAsset, iDebt]
                        if thisDebt > 0:
                            thisFrac = newDebt / thisDebt
                    else:
                        liqDec = 0
                        lowInd, lowWeight = GetLocation(equityvec, thisEquity)
                    for iFType in range(ftNum):
                        for iFE in range(feNum):
                            for iZ in range(zNum2):
                                if liqDec < 1:
                                    thisVal = valfuncFarm[ageInd, iFType, iFE, iZ, iLagK, lowInd] * lowWeight + \
                                              valfuncFarm[ageInd, iFType, iFE, iZ, iLagK, lowInd + 1] * (1 - lowWeight)
                                    if thisVal < liqValue:
                                        liqDec = 1
                                        thisVal = liqValue
                                        thisFrac = 1
                                        newDebt = postliqAssets[iLagK, iTotAsset, iDebt]
                                        if thisDebt > 0:
                                            thisFrac = newDebt / thisDebt

                                liqDecisionMat[ageInd, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = liqDec
                                fracRepaidMat[ageInd, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = thisFrac
                                valfuncMat[ageInd, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = thisVal
                else:
                    for iFType in range(ftNum):
                        for iFE in range(feNum):
                            for iZ in range(zNum2):
                                valfuncMat[ageInd, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = liqValue

                                lowInd, lowWeight = GetLocation(valfuncFarm[ageInd, iFType, iFE, iZ, iLagK], liqValue)

                                thisEquity = equityvec[lowInd] * lowWeight + equityvec[lowInd] * (1 - lowWeight) # GRRRRR. The original behavior is 'equityvec[lowInd + 1]', but this crashes, so w/e!!
                                newDebt = thisAsset - thisEquity
                                liqDec = 0

                                if newDebt > thisDebt:
                                    newDebt = thisDebt
                                    thisEquity = thisAsset - thisDebt

                                    lowInd, lowWeight = GetLocation(equityvec, thisEquity)
                                    contValue = valfuncFarm[ageInd, iFType, iFE, iZ, iLagK, lowInd] * lowWeight + \
                                                valfuncFarm[ageInd, iFType, iFE, iZ, iLagK, lowInd + 1] * (1 - lowWeight)
                                    valfuncMat[ageInd, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = contValue
                                elif newDebt < postliqAssets[iLagK, iTotAsset, iDebt]:
                                    liqDec = 1
                                    newDebt = postliqAssets[iLagK, iTotAsset, iDebt]
                                    if newDebt > thisDebt:
                                        newDebt = thisDebt

                                liqDecisionMat[ageInd, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = liqDec
                                fracRepaidMat[ageInd, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = 1
                                if thisDebt > 0:
                                    fracRepaidMat[ageInd, iFType, iFE, iZ, iLagK, iTotAsset, iDebt] = newDebt / thisDebt

    return liqDecisionMat, fracRepaidMat, valfuncMat


#liqDecisionMat, fracRepaidMat, valfuncMat = getliqDecision(totassetvec, debtvec, equityvec, postliqAssets, postliqNWIndex, valfuncWork, valfuncFarm, liqDecisionMat, valfuncMat, fracRepaidMat, tInd) # This one takes a while ...

# This is basically the function which is scipy.optimize
# We evaluate the entire space, get utility, and if utility is higher than previously recorded, we write down those new indexes.
# We then return about a million indexes
@njit(fastmath=True, parallel=True)
def getOperatingDec(eqAssignvec, equityvec, capitalvec, lagcapvec, debtvec, NKratiovec, cashvec,
                    zTransmtx, CLKindvec, CLKwgtvec, goodGridPoint, NPtotassetWeight, NPtotassetIndex, valfuncMat,
                    fracRepaidVec, valfuncFarm, bestIntRateFarm, bestKIndexFarm, bestNKratIndexFarm, bestCashIndexFarm,
                    bestDividendFarm, bestDebtIndexFarm, bestNPTAWeightFarm, bestNPTAIndexFarm, bestKFarm, bestNKratFarm,
                    bestDebtFarm, bestCashFarm, ageInd):

    minRepay = 0
    if ageInd > (lifespan - 2):
        minRepay = eRepayMin

    stepPct = minStepPct + pow((1 - gridShrinkRate), (lifespan - 1 - ageInd))

    for iFType in range(ftNum):
        #print(f"iFType {iFType} / {ftNum}")
        for iFE in range(feNum):
            for iZ in range(zNum2):
                for iLagK in range(lagcapNum):
                    for jEquity in prange(len(equityvec)):

                        iEquity = eqAssignvec[jEquity]
                        todaysUtility = getUtility(consFloor / 2 - c_0 / 2)
                        continuationValue = valfuncMat[ageInd + 1, iFType, 0, 0, lagcapNum - 1, 0, debtNum - 1]
                        value = todaysUtility + beta * continuationValue
                        maxValue = value - 1e6
                        bestDebtI = -1
                        bestDebt = 0
                        bestIR = bigR
                        bestTotKI = -1
                        bestTotK = 0
                        bestNKratI = -1
                        bestNKrat = 0
                        bestCashI = -1
                        bestCash = 0
                        bestDiv = 0
                        bestNPTAW = bestNPTAWeightFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]
                        bestNPTAI = bestNPTAIndexFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]

                        for iZNP in range(zNum):
                            bestNPTAW[iZNP] = -1
                            bestNPTAI[iZNP] = -1

                        iTotKmin = 0
                        iTotKmax = capitalNum
                        iNKratmin = 0
                        iNKratmax = NKratioNum
                        iCashmin = 0
                        iCashmax = cashNum
                        iDebtmin = 0
                        iDebtmax = debtNum

                        if ageInd < (lifespan - 1) and bestKIndexFarm[ageInd + 1][iFType][iFE][iZ][iLagK][iEquity] > -1:
                            numStepsF = stepPct * capitalNum
                            numSteps = int(numStepsF)
                            if numSteps < 2:
                                numSteps = 2
                            iTotKmin = bestKIndexFarm[ageInd + 1][iFType][iFE][iZ][iLagK][iEquity] - numSteps
                            iTotKmax = iTotKmin + 2 * numSteps + 1
                            if iTotKmin < 0:
                                iTotKmin = 0
                            if iTotKmax > capitalNum:
                                iTotKmax = capitalNum

                            numStepsF = stepPct * NKratioNum
                            numSteps = int(numStepsF)
                            if numSteps < 2:
                                numSteps = 2
                            iNKratmin = bestNKratIndexFarm[ageInd + 1][iFType][iFE][iZ][iLagK][iEquity] - numSteps
                            iNKratmax = iNKratmin + 2 * numSteps + 1
                            if iNKratmin < 0:
                                iNKratmin = 0
                            if iNKratmax > NKratioNum:
                                iNKratmax = NKratioNum

                            numStepsF = stepPct * cashNum
                            numSteps = int(numStepsF)
                            if numSteps < 2:
                                numSteps = 2
                            iCashmin = bestCashIndexFarm[ageInd + 1][iFType][iFE][iZ][iLagK][iEquity] - numSteps
                            iCashmax = iCashmin + 2 * numSteps + 1
                            if iCashmin < 0:
                                iCashmin = 0
                            if iCashmax > cashNum:
                                iCashmax = cashNum

                            numStepsF = stepPct * debtNum
                            numSteps = int(numStepsF)
                            if numSteps < 2:
                                numSteps = 2
                            iDebtmin = bestDebtIndexFarm[ageInd + 1][iFType][iFE][iZ][iLagK][iEquity] - numSteps
                            iDebtmax = iDebtmin + 2 * numSteps + 1
                            if iDebtmin < 0:
                                iDebtmin = 0
                            if iDebtmax > debtNum:
                                iDebtmax = debtNum

                        for iTotK in range(iTotKmin, iTotKmax):
                            salesloss = (1 - delta + eGK) * lagcapvec[iLagK] / bigG - capitalvec[iTotK]
                            if salesloss > 0:
                                salesloss = phi * salesloss
                            else:
                                salesloss = 0

                            for iNKrat in range(iNKratmin, iNKratmax):
                                for iCash in range(iCashmax - 1, iCashmin - 1, -1):
                                    if goodGridPoint[iTotK][iFType][iFE][iNKrat][iCash] == 0:
                                        break

                                    LOA = equityvec[iEquity] - capitalvec[iTotK] - cashvec[iCash] - salesloss
                                    minDebt = -(1 + r_riskfree) * (LOA + c_0 * eqInject) / bigG

                                    for iDebt in range(iDebtmax - 1, iDebtmin - 1, -1):
                                        thisDebt = debtvec[iDebt]
                                        if thisDebt < minDebt:
                                            break
                                        if (thisDebt * psi_inverse) > capitalvec[iTotK]:
                                            continue

                                        eRepay = getExpectation(fracRepaidVec[ageInd + 1][iFType][iFE],
                                                                NPtotassetIndex[iTotK][iFType][iFE][iNKrat][iCash],
                                                                NPtotassetWeight[iTotK][iFType][iFE][iNKrat][iCash],
                                                                CLKindvec[iTotK], CLKwgtvec[iTotK], iDebt,
                                                                zTransmtx[iZ])

                                        if eRepay <= minRepay and thisDebt > 0:
                                            continue
                                        if eRepay > 1:
                                            eRepay = 1
                                        thisLoan = eRepay * thisDebt * bigG / (1 + r_riskfree)
                                        thisDiv = LOA + thisLoan
                                        if thisDiv < (consFloor - c_0 * eqInject):
                                            continue

                                        todaysUtility = getUtility(thisDiv) + chi
                                        continuationValue = getExpectation(valfuncMat[ageInd + 1][iFType][iFE],
                                                                           NPtotassetIndex[iTotK][iFType][iFE][iNKrat][iCash],
                                                                           NPtotassetWeight[iTotK][iFType][iFE][iNKrat][iCash],
                                                                           CLKindvec[iTotK], CLKwgtvec[iTotK], iDebt,
                                                                           zTransmtx[iZ])
                                        value = todaysUtility + beta * continuationValue

                                        if value > maxValue:
                                            maxValue = value
                                            bestDebtI = iDebt
                                            if debtvec[iDebt] == 0:
                                                eRepay = 1
                                            bestIR = (1 + r_riskfree) / eRepay
                                            bestTotKI = iTotK
                                            bestNKratI = iNKrat
                                            bestCashI = iCash
                                            bestDiv = thisDiv
                                            bestTotK = capitalvec[iTotK]
                                            bestNKrat = NKratiovec[iNKrat]
                                            bestDebt = debtvec[iDebt]
                                            bestCash = cashvec[iCash]

                        valfuncFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = maxValue
                        bestCashIndexFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestCashI
                        bestDebtIndexFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestDebtI
                        bestIntRateFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestIR
                        bestKIndexFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestTotKI
                        bestNKratIndexFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestNKratI
                        bestDividendFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestDiv
                        bestKFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestTotK
                        bestNKratFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestNKrat
                        bestDebtFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestDebt
                        bestCashFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestCash

                        if bestTotKI > -1:
                            for iZNP in range(zNum):
                                bestNPTAW[iZNP] = NPtotassetWeight[bestTotKI][iFType][iFE][bestNKratI][bestCashI][iZNP]
                                bestNPTAI[iZNP] = NPtotassetIndex[bestTotKI][iFType][iFE][bestNKratI][bestCashI][iZNP]

    return valfuncFarm, bestCashIndexFarm, bestDebtIndexFarm, bestIntRateFarm, bestKIndexFarm, bestNKratIndexFarm, bestDividendFarm, bestKFarm, bestNKratFarm, bestDebtFarm, bestCashFarm

_="""
equityAssignments = getAssignmentVec(equityNum, 1);

valfuncFarm, bestCashIndexFarm, bestDebtIndexFarm, bestIntRateFarm, bestKIndexFarm, bestNKratIndexFarm, bestDividendFarm, bestKFarm, bestNKratFarm, bestDebtFarm, bestCashFarm = getOperatingDec(equityAssignments, tInd, equityvec, capitalvec,
                          lagcapvec, debtvec, NKratiovec, cashvec, zTransmtx, CLKindvec, CLKwgtvec,
                          goodGridPoint, NPtotassetWeight, NPtotassetIndex, valfuncMat, fracRepaidMat,
                          valfuncFarm, bestIntRateFarm, bestKIndexFarm, bestNKratIndexFarm,
                          bestCashIndexFarm, bestDividendFarm ,bestDebtIndexFarm, bestNPTAWeightFarm,
                          bestNPTAIndexFarm, bestKFarm, bestNKratFarm, bestDebtFarm, bestCashFarm)
"""



### THE BIG LOOP

#lifespan = 2

size = 1
equityAssignments = getAssignmentVec(equityNum, size);


# We structure this into tuples so that, here at the end, we can package the entire thing into *FAST*
# To put in tInd, it just needs to go at either the front or the back of the argument list

# get operating decision
getOperatingDec_input = equityAssignments, equityvec, capitalvec, lagcapvec, debtvec, NKratiovec, cashvec, zTransmtx, CLKindvec, CLKwgtvec, goodGridPoint, NPtotassetWeight, NPtotassetIndex, valfuncMat, fracRepaidMat, valfuncFarm, bestIntRateFarm, bestKIndexFarm, bestNKratIndexFarm, bestCashIndexFarm, bestDividendFarm ,bestDebtIndexFarm, bestNPTAWeightFarm, bestNPTAIndexFarm, bestKFarm, bestNKratFarm, bestDebtFarm, bestCashFarm
getOperatingDec_output = valfuncFarm, bestCashIndexFarm, bestDebtIndexFarm, bestIntRateFarm, bestKIndexFarm, bestNKratIndexFarm, bestDividendFarm, bestKFarm, bestNKratFarm, bestDebtFarm, bestCashFarm

# liquidity farm
getliqDecision_input = totassetvec, debtvec, equityvec, postliqAssets, postliqNWIndex, valfuncWork, valfuncFarm, liqDecisionMat, valfuncMat, fracRepaidMat
getLiqDecision_output = liqDecisionMat, fracRepaidMat, valfuncMat

#print(getliqDecision_input)

@njit
def main_loop(input_op, output_op, input_liq, output_liq) :
  for tInd in range(lifespan - 1, -1, -1):
      print(tInd)

      output_op = getOperatingDec(*input_op, tInd)

      output_liq = getliqDecision(*input_liq, tInd)

  return output_op, output_liq

output_final_op, output_final_liq = main_loop(getOperatingDec_input, getOperatingDec_output, getliqDecision_input, getLiqDecision_output) 

# Den dropper dem her lige nu i dit default usr/directory. Oh well!
np.save("feValues", feValues)
np.save("feValues", feValues)
np.save("zValues", zValues)
np.save("totassetvec", totassetvec)
np.save("debtvec", debtvec)
np.save("equityvec", equityvec)
np.save("cashvec", cashvec)
np.save("lagcapvec", lagcapvec)
np.save("liqDecisionMat", liqDecisionMat)
np.save("fracRepaidMat", fracRepaidMat)
np.save("bestIntRateFarm", bestIntRateFarm)
np.save("bestCashFarm", bestCashFarm)
np.save("bestDividendFarm", bestDividendFarm)
np.save("bestKFarm", bestKFarm)
np.save("bestNKratFarm", bestNKratFarm)
np.save("bestDebtFarm", bestDebtFarm)

"""
TODO FOR IMPLEMENTATION.


// All this write stuff we might not need.
void WriteFunctions(double **valfuncWork, double **bestCWork, double **bestNPIWork,
                    double ******valfuncFarm, double ******bestIntRateFarm, double ******bestCashFarm,
                    double ******bestKFarm, double ******bestNKratFarm, double ******bestDividendFarm,
                    double ******bestDebtFarm, double *******liqDecisionMat, double *******valfuncMat,
                    double *******fracRepaidMat, double *assetvec, double *equityvec, double *lagcapvec,
                    double *debtvec, double *totassetvec,  double *feValues);
void WriteSims(double **FEIsimsMtx, double **ZsimsMtx, double **ZIsimsMtx, double **asstsimsMtx,
               double **dividendsimsMtx, double **totKsimsMtx, double **NKratsimsMtx, double **cashsimsMtx,
               double **IRsimsMtx, double **debtsimsMtx, double **NWsimsMtx, double **fracRepaidsimsMtx,
               double **outputsimsMtx, double **liqDecsimsMtx, double **agesimsMtx, double **expensesimsMtx);


// Important
double intrplte7D(double *******decruleMat, int ageInd, int ftInd, int feInd, int zInd2, int lkInd2, int taInd, int dInd,
                  double dWgt, double taWgt, double feWgt, double zWgt, double lkWgt);
double intrplte6D(double ******decruleMat, int ageInd, int ftInd, int feInd, int zInd2, int lkInd2, int eqInd,
                  double eqWgt, double feWgt, double zWgt, double lkWgt);

// STØRSTE FUNKTIONER. DET SIDSTE VI TAGER.

void simulation(double *initAges, double *initYears, double *initCapital, double *initTotAssets,
                double *initDebt, double *farmtypes, double *feShksVec, double *feValues, double **zShksMtx,
                double *zValues, double *totassetvec, double *debtvec, double *equityvec, double *cashvec,
                double *lagcapvec, double **FEIsimsMtx, double **ZsimsMtx, double **ZIsimsMtx,
                double **asstsimsMtx, double **dividendsimsMtx, double **totKsimsMtx, double **NKratsimsMtx,
                double **cashsimsMtx, double **IRsimsMtx, double **debtsimsMtx, double **NWsimsMtx,
                double **fracRepaidsimsMtx, double **outputsimsMtx, double **liqDecsimsMtx, double **agesimsMtx,
                double **expensesimsMtx, double *******liqDecisionMat, double *******fracRepaidMat,
                double ******bestIntRateFarm, double ******bestCashFarm, double ******bestDividendFarm,
                double ******bestKFarm, double ******bestNKratFarm, double ******bestDebtFarm,
                int iSimmin, int iSimmax);
"""

feValues = np.load(VFI_path + "feValues.npy")
zValues = np.load(VFI_path + "zValues.npy")
totassetvec = np.load(VFI_path + "totassetvec.npy")
debtvec = np.load(VFI_path + "debtvec.npy")
equityvec = np.load(VFI_path + "equityvec.npy")
cashvec = np.load(VFI_path + "cashvec.npy")
lagcapvec = np.load(VFI_path + "lagcapvec.npy")
liqDecisionMat = np.load(VFI_path + "liqDecisionMat.npy")
fracRepaidMat = np.load(VFI_path + "fracRepaidMat.npy")
bestIntRateFarm = np.load(VFI_path + "bestIntRateFarm.npy")
bestCashFarm = np.load(VFI_path + "bestCashFarm.npy")
bestDividendFarm = np.load(VFI_path + "bestDividendFarm.npy")
bestKFarm = np.load(VFI_path + "bestKFarm.npy")
bestNKratFarm = np.load(VFI_path + "bestNKratFarm.npy")
bestDebtFarm = np.load(VFI_path + "bestDebtFarm.npy")

# Load the simulation files using np.loadtxt
feShks = np.loadtxt(iopath + "feshks.txt")
FTypesims = np.loadtxt(iopath + "ftype_sim.txt")
initAges = np.loadtxt(iopath + "initages.txt")
initDebt = np.loadtxt(iopath + "initdebt.txt")
initCapital = np.loadtxt(iopath + "initK.txt")
initTotAssets = np.loadtxt(iopath + "initta.txt")
initYears = np.loadtxt(iopath + "inityrs.txt")
zShks = np.loadtxt(iopath + "zshks.txt")

recsize = (timespan + 1) * numSims

# Initialize arrays with NaN values
Zsims = np.nan + np.zeros(recsize)
ZIsims = np.nan + np.zeros(recsize)  # Index numbers, TFP shock
FEIsims = np.nan + np.zeros(recsize)  # Index numbers, fixed Effect TFP shock
asstsims = np.nan + np.zeros(recsize)  # Total Assets, beginning of period
debtsims = np.nan + np.zeros(recsize)  # Debt, beginning of period, pre-renegotiation
fracRepaidsims = np.nan + np.zeros(recsize)  # Fraction of outstanding debt repaid
liqDecsims = np.nan + np.zeros(recsize)  # Liquidation decisions
agesims = np.nan + np.zeros(recsize)  # Age of farm head
dividendsims = np.nan + np.zeros(recsize)  # Dividends/consumption
totKsims = np.nan + np.zeros(recsize)  # Capital Stock, beginning of period
NKratsims = np.nan + np.zeros(recsize)  # igoods/capital ratio
cashsims = np.nan + np.zeros(recsize)  # Cash/liquid assets
IRsims = np.nan + np.zeros(recsize)  # Contractual interest rates
NWsims = np.nan + np.zeros(recsize)  # Net worth for period, post-renegotiation
outputsims = np.nan + np.zeros(recsize)  # Output/revenues
expensesims = np.nan + np.zeros(recsize)  # Operating expenditures

# Reshape the arrays
zShksMtx = zShks.reshape(timespan + 2, numSims+24)
ZsimsMtx = Zsims.reshape(timespan + 1, numSims)
ZIsimsMtx = ZIsims.reshape(timespan + 1, numSims)
FEIsimsMtx = FEIsims.reshape(timespan + 1, numSims)
asstsimsMtx = asstsims.reshape(timespan + 1, numSims)
debtsimsMtx = debtsims.reshape(timespan + 1, numSims)
fracRepaidsimsMtx = fracRepaidsims.reshape(timespan + 1, numSims)
liqDecsimsMtx = liqDecsims.reshape(timespan + 1, numSims)
agesimsMtx = agesims.reshape(timespan + 1, numSims)
dividendsimsMtx = dividendsims.reshape(timespan + 1, numSims)
totKsimsMtx = totKsims.reshape(timespan + 1, numSims)
NKratsimsMtx = NKratsims.reshape(timespan + 1, numSims)
cashsimsMtx = cashsims.reshape(timespan + 1, numSims)
IRsimsMtx = IRsims.reshape(timespan + 1, numSims)
NWsimsMtx = NWsims.reshape(timespan + 1, numSims)
outputsimsMtx = outputsims.reshape(timespan + 1, numSims)
expensesimsMtx = expensesims.reshape(timespan + 1, numSims)

def intrplte6D(decruleMat, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd,
			   eqWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum):
	interpVal = (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][eqInd] * eqWgt +
				 decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][eqInd + 1] * (1 - eqWgt)) * feWgt

	if feNum > 1:
		interpVal += (decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2][eqInd] * eqWgt +
					  decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2][eqInd + 1] * (1 - eqWgt)) * (1 - feWgt)

	if zNum2 > 1:
		tempVal0 = (decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2][eqInd] * eqWgt +
					decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2][eqInd + 1] * (1 - eqWgt)) * feWgt

		if feNum > 1:
			tempVal0 += (decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2][eqInd] * eqWgt +
						 decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2][eqInd + 1] * (1 - eqWgt)) * (1 - feWgt)

		interpVal = interpVal * zWgt + tempVal0 * (1 - zWgt)

	if lagcapNum > 2:
		tempVal1 = (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2 + 1][eqInd] * eqWgt +
					decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2 + 1][eqInd + 1] * (1 - eqWgt)) * feWgt

		if feNum > 1:
			tempVal1 += (decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2 + 1][eqInd] * eqWgt +
						 decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2 + 1][eqInd + 1] * (1 - eqWgt)) * (1 - feWgt)

		if zNum2 > 1:
			tempVal0 = (decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2 + 1][eqInd] * eqWgt +
						decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2 + 1][eqInd + 1] * (1 - eqWgt)) * feWgt

			if feNum > 1:
				tempVal0 += (decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][eqInd] * eqWgt +
							 decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][eqInd + 1] * (1 - eqWgt)) * (
										1 - feWgt)

			tempVal1 = tempVal1 * zWgt + tempVal0 * (1 - zWgt)

		interpVal = interpVal * lkWgt + tempVal1 * (1 - lkWgt)

	return interpVal


def intrplte7D(decruleMat, ageInd, ftInd, feInd, zInd2, lkInd2, taInd, dInd,
			   dWgt, taWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum):
	interpVal = ((decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd][dInd] * dWgt +
				  decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd][dInd + 1] * (1 - dWgt)) * taWgt +
				 (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd + 1][dInd] * dWgt +
				  decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd + 1][dInd + 1] * (1 - dWgt)) * (
							 1 - taWgt)) * feWgt

	if feNum > 1:
		interpVal += ((decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2][taInd][dInd] * dWgt +
					   decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2][taInd][dInd + 1] * (1 - dWgt)) * taWgt +
					  (decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2][taInd + 1][dInd] * dWgt +
					   decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2][taInd + 1][dInd + 1] * (1 - dWgt)) * (
								  1 - taWgt)) * (1 - feWgt)

	if zNum2 > 1:
		tempVal0 = ((decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2][taInd][dInd] * dWgt +
					 decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2][taInd][dInd + 1] * (1 - dWgt)) * taWgt +
					(decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2][taInd + 1][dInd] * dWgt +
					 decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2][taInd + 1][dInd + 1] * (1 - dWgt)) * (
								1 - taWgt)) * feWgt

		if feNum > 1:
			tempVal0 += ((decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2][taInd][dInd] * dWgt +
						  decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2][taInd][dInd + 1] * (
									  1 - dWgt)) * taWgt +
						 (decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2][taInd + 1][dInd] * dWgt +
						  decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2][taInd + 1][dInd + 1] * (1 - dWgt)) * (
									 1 - taWgt)) * (1 - feWgt)

		interpVal = interpVal * zWgt + tempVal0 * (1 - zWgt)

	if lagcapNum > 2:
		tempVal1 = ((decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2 + 1][taInd][dInd] * dWgt +
					 decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2 + 1][taInd][dInd + 1] * (1 - dWgt)) * taWgt +
					(decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2 + 1][taInd + 1][dInd] * dWgt +
					 decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2 + 1][taInd + 1][dInd + 1] * (1 - dWgt)) * (
								1 - taWgt)) * feWgt

		if feNum > 1:
			tempVal1 += ((decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2 + 1][taInd][dInd] * dWgt +
						  decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2 + 1][taInd][dInd + 1] * (
									  1 - dWgt)) * taWgt +
						 (decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2 + 1][taInd + 1][dInd] * dWgt +
						  decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2 + 1][taInd + 1][dInd + 1] * (1 - dWgt)) * (
									 1 - taWgt)) * (1 - feWgt)

		if zNum2 > 1:
			tempVal0 = ((decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2 + 1][taInd][dInd] * dWgt +
						 decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2 + 1][taInd][dInd + 1] * (
									 1 - dWgt)) * taWgt +
						(decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2 + 1][taInd + 1][dInd] * dWgt +
						 decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2 + 1][taInd + 1][dInd + 1] * (1 - dWgt)) * (
									1 - taWgt)) * feWgt

			if feNum > 1:
				tempVal0 += ((decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][taInd][dInd] * dWgt +
							  decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][taInd][dInd + 1] * (
										  1 - dWgt)) * taWgt +
							 (decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][taInd + 1][dInd] * dWgt +
							  decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][taInd + 1][dInd + 1] * (
										  1 - dWgt)) * (1 - taWgt)) * (1 - feWgt)

			tempVal1 = tempVal1 * zWgt + tempVal0 * (1 - zWgt)

		interpVal = interpVal * lkWgt + tempVal1 * (1 - lkWgt)

	return interpVal

def simulation(initAges, initYears, initCapital, initTotAssets,
                 initDebt, farmtypes, feShksVec, feValues, zShksMtx, zValues,
                 totassetvec, debtvec,  equityvec, cashvec, lagcapvec, FEIsimsMtx, ZsimsMtx, ZIsimsMtx,
                 asstsimsMtx, dividendsimsMtx, totKsimsMtx, NKratsimsMtx, cashsimsMtx, IRsimsMtx,
                 debtsimsMtx, NWsimsMtx, fracRepaidsimsMtx, outputsimsMtx,liqDecsimsMtx, agesimsMtx,
                 expensesimsMtx, liqDecisionMat, fracRepaidMat, bestIntRateFarm, bestCashFarm,
                 bestDividendFarm, bestKFarm, bestNKratFarm, bestDebtFarm, numSims):
  """ Timing: at the begining of each period, know total assets, total debt.
      Next, decide whether to operate.  Exiting farmers is an absorbing state.
      Finally, pick operating decisions for this period """

  print(f"Rank=0, numSims={numSims}")

  for personInd in range(numSims): # loop for farms
      age0 = int(initAges[personInd])
      year0 = int(initYears[personInd]) + 1 # GAUSS indexing

      ftInd = 0
      if ftNum > 1:
          ftInd = int(farmtypes[personInd])- 1 # GAUSS indexing

      if ftInd == 0:
          alpha, gamma, ag2, ag3, gag, agag = prodFnParms(alpha1, gamma1)
      else:
          alpha, gamma, ag2, ag3, gag, agag = prodFnParms(alpha2, gamma2)

      feValue = np.exp(feShksVec[personInd])
      feInd, feWgt = GetLocation(feValues, feValue)
      if feNum == 1:
        feWgt = 1

      FEIsimsMtx[0][personInd] = feInd + 1 # GAUSS indexing
      if feWgt < 0.5:
          FEIsimsMtx[0][personInd] += 1

      for yearInd in range(year0): # loop of year
          dividendsimsMtx[yearInd][personInd] = -1e5
          totKsimsMtx[yearInd][personInd] = -1
          NKratsimsMtx[yearInd][personInd] = -1
          IRsimsMtx[yearInd][personInd] = -1
          NWsimsMtx[yearInd][personInd] = -1e5
          expensesimsMtx[yearInd][personInd] = -1
          outputsimsMtx[yearInd][personInd] = -1
          cashsimsMtx[yearInd][personInd] = -1
          asstsimsMtx[yearInd][personInd] = -1e5
          debtsimsMtx[yearInd][personInd] = -1e5
          fracRepaidsimsMtx[yearInd + 1][personInd] = -1
          liqDecsimsMtx[yearInd][personInd] = -1
          agesimsMtx[yearInd][personInd] = -1

          zValue = np.exp(zShksMtx[yearInd][personInd]) # next periods's shock
          zInd, zWgt = GetLocation(zValues, zValue)
          ZsimsMtx[yearInd][personInd] = zValue
          ZIsimsMtx[yearInd][personInd] = zInd + 1 # GAUSS indexing
          if zWgt < 0.5:
              ZIsimsMtx[yearInd + 1][personInd] += 1

      age = age0
      ageInd = age - bornage
      if ageInd < 0:
          ageInd = 0

      zValue = np.exp(zShksMtx[year0][personInd])
      zInd, zWgt = GetLocation(zValues, zValue)
      zInd2 = 0
      if zNum2 > 1:
          zInd2 = zInd

      agesimsMtx[year0][personInd] = float(age)
      ZsimsMtx[year0][personInd] = zValue
      ZIsimsMtx[year0][personInd] = zInd + 1 # GAUSS indexing
      if zWgt < 0.5:
          ZIsimsMtx[year0][personInd] += 1

      lagCapital = initCapital[personInd]
      totAssets = initTotAssets[personInd]
      debt = initDebt[personInd]
      lkInd, lkWgt = GetLocation(lagcapvec, lagCapital)
      lkInd2 = 0
      if lagcapNum > 1:
          lkInd2 = lkInd
      taInd, taWgt = GetLocation(totassetvec, totAssets)
      dInd, dWgt = GetLocation(debtvec, debt)
      liqDec = 0
      fracRepaid = 1

      liqDec_dbl = intrplte7D(liqDecisionMat, ageInd, ftInd, feInd, zInd2, lkInd2, taInd, dInd, dWgt, taWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum)
      fracRepaid = intrplte7D(fracRepaidMat, ageInd, ftInd, feInd, zInd2, lkInd2, taInd, dInd, dWgt, taWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum)
      if liqDec_dbl > 0.5:
          liqDec = 1
      if fracRepaid > 1:
          fracRepaid = 1
      if fracRepaid < 0:
          fracRepaid = 0

      asstsimsMtx[year0][personInd] = totAssets
      debtsimsMtx[year0][personInd] = debt
      fracRepaidsimsMtx[year0][personInd] = fracRepaid
      liqDecsimsMtx[year0][personInd] = float(liqDec)

      for yearInd in range(year0, timespan + 1):
          age = age0 + yearInd - year0
          ageInd = age - bornage
          if ageInd < 0:
              ageInd = 0
          agesimsMtx[yearInd][personInd] = float(age)

          if liqDec == 1:
              dividendsimsMtx[yearInd][personInd] = -1
              totKsimsMtx[yearInd][personInd] = -1
              NKratsimsMtx[yearInd][personInd] = -1
              IRsimsMtx[yearInd][personInd] = -1
              NWsimsMtx[yearInd][personInd] = -1
              expensesimsMtx[yearInd][personInd] = -1
              outputsimsMtx[yearInd][personInd] = -1
              cashsimsMtx[yearInd][personInd] = -1
              if yearInd == timespan:
                  continue

              asstsimsMtx[yearInd + 1][personInd] = -1e5
              debtsimsMtx[yearInd + 1][personInd] = -1
              fracRepaidsimsMtx[yearInd + 1][personInd] = -1
              liqDecsimsMtx[yearInd + 1][personInd] = 1

              zValue = np.exp(zShksMtx[yearInd + 1][personInd])
              zInd, zWgt = GetLocation(zValues, zValue)
              ZsimsMtx[yearInd + 1][personInd] = zValue
              ZIsimsMtx[yearInd + 1][personInd] = zInd + 1 # GAUSS indexing
              if zWgt < 0.5:
                  ZIsimsMtx[yearInd + 1][personInd] += 1
              continue

          equity = totAssets - fracRepaid * debt
          eqInd, eqWgt = GetLocation(equityvec, equity)

          totK = intrplte6D(bestKFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, eqWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum)
          NKrat = intrplte6D(bestNKratFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, eqWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum)
          intRate = intrplte6D(bestIntRateFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, eqWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum)
          cash = intrplte6D(bestCashFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, eqWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum)
          debt = intrplte6D(bestDebtFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, eqWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum)

          if totK < 0:
              totK = 0
          if NKrat < 0:
              NKrat = 0
          if intRate < bigR:
              intRate = bigR
          if cash < 0:
              cash = 0
          if debt < 0:
              debt = 0

          dividend = equity + debt / intRate - totK - cash
          if dividend < (-c_0 * eqInject):
              dividend = (-c_0 * eqInject)

          igoods = getBaseIGoods(totK, feValue, gag, agag, ag3, igshift) * NKrat
          expenses = igoods + fixedcost
          zValue = np.exp(zShksMtx[yearInd + 1][personInd]) # next period's shock
          output = feValue * zValue * getBaseRevenues(totK, igoods, gamma, ag2) # output = revenues

          dividendsimsMtx[yearInd][personInd] = dividend
          totKsimsMtx[yearInd][personInd] = totK
          NKratsimsMtx[yearInd][personInd] = NKrat
          NWsimsMtx[yearInd][personInd] = equity
          expensesimsMtx[yearInd][personInd] = expenses
          outputsimsMtx[yearInd][personInd] = output
          IRsimsMtx[yearInd][personInd] = intRate
          cashsimsMtx[yearInd][personInd] = cash

          if yearInd == timespan:
              continue

          # now move to t+1 states

          lagCapital = totK
          totAssets = ((1 - delta + eGK) * totK + output - expenses + cash) / bigG
          taInd, taWgt = GetLocation(totassetvec, totAssets)
          dInd, dWgt = GetLocation(debtvec, debt)
          lkInd, lkWgt = GetLocation(lagcapvec, lagCapital)
          lkInd2 = 0
          if lagcapNum > 1:
              lkInd2 = lkInd

          zInd, zWgt = GetLocation(zValues, zValue)
          zInd2 = 0
          if zNum2 > 1:
              zInd2 = zInd
          ZsimsMtx[yearInd + 1][personInd] = zValue
          ZIsimsMtx[yearInd + 1][personInd] = zInd # + 1 GAUSS indexing
          if zWgt < 0.5:
              ZIsimsMtx[yearInd + 1][personInd] += 1
          FEIsimsMtx[yearInd + 1][personInd] = FEIsimsMtx[0][personInd]

          liqDec = 0
          fracRepaid = 1

          liqDec_dbl = intrplte7D(liqDecisionMat, ageInd + 1, ftInd, feInd, zInd2, lkInd2, taInd, dInd, dWgt, taWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum)
          fracRepaid = intrplte7D(fracRepaidMat, ageInd + 1, ftInd, feInd, zInd2, lkInd2, taInd, dInd, dWgt, taWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum)

          if liqDec_dbl > 0.5:
              liqDec = 1
          if fracRepaid > 1:
              fracRepaid = 1
          if fracRepaid < 0:
              fracRepaid = 0

          asstsimsMtx[yearInd + 1][personInd] = totAssets
          debtsimsMtx[yearInd + 1][personInd] = debt
          fracRepaidsimsMtx[yearInd + 1][personInd] = fracRepaid
          liqDecsimsMtx[yearInd + 1][personInd] = float(liqDec)

simulation(initAges, initYears, initCapital, initTotAssets,
                 initDebt, FTypesims, feShks, feValues, zShksMtx, zValues,
                 totassetvec, debtvec,  equityvec, cashvec, lagcapvec, FEIsimsMtx, ZsimsMtx, ZIsimsMtx,
                 asstsimsMtx, dividendsimsMtx, totKsimsMtx, NKratsimsMtx, cashsimsMtx, IRsimsMtx,
                 debtsimsMtx, NWsimsMtx, fracRepaidsimsMtx, outputsimsMtx,liqDecsimsMtx, agesimsMtx,
                 expensesimsMtx, liqDecisionMat, fracRepaidMat, bestIntRateFarm, bestCashFarm,
                 bestDividendFarm, bestKFarm, bestNKratFarm, bestDebtFarm, numSims)

def save_processed_data_txt(data, filename, savepath):
    """
    Save processed data to a text file.

    Parameters:
    data: The data to save (numpy array).
    filename: The name of the file.
    savepath: The path to the directory where the file will be saved.
    """
    filepath = os.path.join(savepath, filename)
    if isinstance(data, np.ndarray):
        np.savetxt(filepath, data)
    else:
        raise ValueError("Data must be a numpy array.")
    print(f"Saved {filename} to {savepath}")

save_processed_data_txt(FEIsimsMtx, "FEindxS.txt", iopath)
save_processed_data_txt(ZsimsMtx, "ZValsS.txt", iopath)
save_processed_data_txt(ZIsimsMtx, "ZindxS.txt", iopath)
save_processed_data_txt(asstsimsMtx, "assetsS.txt", iopath)
save_processed_data_txt(debtsimsMtx, "debtS.txt", iopath)
save_processed_data_txt(fracRepaidsimsMtx, "fracRPS.txt", iopath)
save_processed_data_txt(liqDecsimsMtx, "liqDecS.txt", iopath)
save_processed_data_txt(agesimsMtx, "ageS.txt", iopath)
save_processed_data_txt(dividendsimsMtx, "divS.txt", iopath)
save_processed_data_txt(totKsimsMtx, "totKS.txt", iopath)
save_processed_data_txt(NKratsimsMtx, "NKratos.txt", iopath)
save_processed_data_txt(cashsimsMtx, "cashS.txt", iopath)
save_processed_data_txt(IRsimsMtx, "intRateS.txt", iopath)
save_processed_data_txt(NWsimsMtx, "equityS.txt", iopath)
save_processed_data_txt(outputsimsMtx, "outputS.txt", iopath)
save_processed_data_txt(expensesimsMtx, "expenseS.txt", iopath)
