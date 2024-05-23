iopath = "C:\\Users\\jomo0\\OneDrive\\Skrivebord\\dp_copy\\estimation_fake\\iofiles\\"

# Other imports
from typing import List, Tuple, Final # Try to statically type everything so we know wtf is going on.
import numpy as np # brug numpy så meget som muligt.
import math
import numpy.typing as npt
from tools import *

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
  alpha: float = alpha0;
  gamma: float = gamma0;
  ag2: float   = 1 - alpha - gamma;
  ag3: float   = 1 / (alpha + gamma);
  gag: float   = (gamma)*(ag3);
  agag: float  = pow(ag2, ag3);

  return alpha, gamma, ag2, ag3, gag, agag

alpha, gamma, ag2, ag3, gag, agag = prodFnParms(alpha1, gamma1)

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

print(getBaseIGoods(100.0, 1.0, 1.0, 1.0, 1.0, 1.0))

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

sim_path = "C:\\Users\\jomo0\\OneDrive\\Skrivebord\\dp_upch\\simulation inputs\\"

# Load the simulation files using np.loadtxt
feShks = np.loadtxt(sim_path + "feshks.txt")
FTypesims = np.loadtxt(sim_path + "ftype_sim.txt")
initAges = np.loadtxt(sim_path + "initages.txt")
initDebt = np.loadtxt(sim_path + "initdebt.txt")
initCapital = np.loadtxt(sim_path + "initK.txt")
initTotAssets = np.loadtxt(sim_path + "initta.txt")
initYears = np.loadtxt(sim_path + "inityrs.txt")
zShks = np.loadtxt(sim_path + "zshks.txt")

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

folder_path = "C:\\Users\\jomo0\\OneDrive\\Skrivebord\\dp_upch\\VFI output\\"

feValues = np.load(folder_path + "feValues.npy")
zValues = np.load(folder_path + "zValues.npy")
totassetvec = np.load(folder_path + "totassetvec.npy")
debtvec = np.load(folder_path + "debtvec.npy")
equityvec = np.load(folder_path + "equityvec.npy")
cashvec = np.load(folder_path + "cashvec.npy")
lagcapvec = np.load(folder_path + "lagcapvec.npy")
liqDecisionMat = np.load(folder_path + "liqDecisionMat.npy")
fracRepaidMat = np.load(folder_path + "fracRepaidMat.npy")
bestIntRateFarm = np.load(folder_path + "bestIntRateFarm.npy")
bestCashFarm = np.load(folder_path + "bestCashFarm.npy")
bestDividendFarm = np.load(folder_path + "bestDividendFarm.npy")
bestKFarm = np.load(folder_path + "bestKFarm.npy")
bestNKratFarm = np.load(folder_path + "bestNKratFarm.npy")
bestDebtFarm = np.load(folder_path + "bestDebtFarm.npy")

sim_path = "C:\\Users\\jomo0\\OneDrive\\Skrivebord\\dp_upch\\simulation inputs\\"

# Load the simulation files using np.loadtxt
feShks = np.loadtxt(sim_path + "feshks.txt")
FTypesims = np.loadtxt(sim_path + "ftype_sim.txt")
initAges = np.loadtxt(sim_path + "initages.txt")
initDebt = np.loadtxt(sim_path + "initdebt.txt")
initCapital = np.loadtxt(sim_path + "initK.txt")
initTotAssets = np.loadtxt(sim_path + "initta.txt")
initYears = np.loadtxt(sim_path + "inityrs.txt")
zShks = np.loadtxt(sim_path + "zshks.txt")

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
          ftInd = int(farmtypes[personInd]) - 1 # GAUSS indexing

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
          ZIsimsMtx[yearInd][personInd] = zInd  + 1 #GAUSS indexing
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
      ZIsimsMtx[year0][personInd] = zInd  + 1 # GAUSS indexing
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
          if zNum > 1:
              zInd2 = zInd
          ZsimsMtx[yearInd + 1][personInd] = zValue
          ZIsimsMtx[yearInd + 1][personInd] = zInd + 1 # GAUSS indexing
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

@jit(nopython=True)
def Locate(Xarray: npt.ArrayLike, x: float) -> int:
    idx = np.searchsorted(Xarray, x, side='left') - 1

    return max(idx, 0) # Bounds our index to the left

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
                             decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][eqInd + 1] * (1 - eqWgt)) * (1 - feWgt)

            tempVal1 = tempVal1 * zWgt + tempVal0 * (1 - zWgt)
        
        interpVal = interpVal * lkWgt + tempVal1 * (1 - lkWgt)
    
    return interpVal

def intrplte7D(decruleMat, ageInd, ftInd, feInd, zInd2, lkInd2, taInd, dInd,
               dWgt, taWgt, feWgt, zWgt, lkWgt, feNum, zNum2, lagcapNum):
    
    interpVal = ((decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd][dInd] * dWgt + 
                  decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd][dInd + 1] * (1 - dWgt)) * taWgt + 
                 (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd + 1][dInd] * dWgt + 
                  decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd + 1][dInd + 1] * (1 - dWgt)) * (1 - taWgt)) * feWgt
    
    if feNum > 1:
        interpVal += ((decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2][taInd][dInd] * dWgt + 
                       decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2][taInd][dInd + 1] * (1 - dWgt)) * taWgt + 
                      (decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2][taInd + 1][dInd] * dWgt + 
                       decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2][taInd + 1][dInd + 1] * (1 - dWgt)) * (1 - taWgt)) * (1 - feWgt)
    
    if zNum2 > 1:
        tempVal0 = ((decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2][taInd][dInd] * dWgt + 
                     decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2][taInd][dInd + 1] * (1 - dWgt)) * taWgt + 
                    (decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2][taInd + 1][dInd] * dWgt + 
                     decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2][taInd + 1][dInd + 1] * (1 - dWgt)) * (1 - taWgt)) * feWgt
        
        if feNum > 1:
            tempVal0 += ((decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2][taInd][dInd] * dWgt + 
                          decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2][taInd][dInd + 1] * (1 - dWgt)) * taWgt + 
                         (decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2][taInd + 1][dInd] * dWgt + 
                          decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2][taInd + 1][dInd + 1] * (1 - dWgt)) * (1 - taWgt)) * (1 - feWgt)
        
        interpVal = interpVal * zWgt + tempVal0 * (1 - zWgt)
    
    if lagcapNum > 2:
        tempVal1 = ((decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2 + 1][taInd][dInd] * dWgt + 
                     decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2 + 1][taInd][dInd + 1] * (1 - dWgt)) * taWgt + 
                    (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2 + 1][taInd + 1][dInd] * dWgt + 
                     decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2 + 1][taInd + 1][dInd + 1] * (1 - dWgt)) * (1 - taWgt)) * feWgt
        
        if feNum > 1:
            tempVal1 += ((decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2 + 1][taInd][dInd] * dWgt + 
                          decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2 + 1][taInd][dInd + 1] * (1 - dWgt)) * taWgt + 
                         (decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2 + 1][taInd + 1][dInd] * dWgt + 
                          decruleMat[ageInd][ftInd][feInd + 1][zInd2][lkInd2 + 1][taInd + 1][dInd + 1] * (1 - dWgt)) * (1 - taWgt)) * (1 - feWgt)
        
        if zNum2 > 1:
            tempVal0 = ((decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2 + 1][taInd][dInd] * dWgt + 
                         decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2 + 1][taInd][dInd + 1] * (1 - dWgt)) * taWgt + 
                        (decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2 + 1][taInd + 1][dInd] * dWgt + 
                         decruleMat[ageInd][ftInd][feInd][zInd2 + 1][lkInd2 + 1][taInd + 1][dInd + 1] * (1 - dWgt)) * (1 - taWgt)) * feWgt
            
            if feNum > 1:
                tempVal0 += ((decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][taInd][dInd] * dWgt + 
                              decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][taInd][dInd + 1] * (1 - dWgt)) * taWgt + 
                             (decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][taInd + 1][dInd] * dWgt + 
                              decruleMat[ageInd][ftInd][feInd + 1][zInd2 + 1][lkInd2 + 1][taInd + 1][dInd + 1] * (1 - dWgt)) * (1 - taWgt)) * (1 - feWgt)
            
            tempVal1 = tempVal1 * zWgt + tempVal0 * (1 - zWgt)
        
        interpVal = interpVal * lkWgt + tempVal1 * (1 - lkWgt)
    
    return interpVal

simulation(initAges, initYears, initCapital, initTotAssets,
                 initDebt, FTypesims, feShks, feValues, zShksMtx, zValues,
                 totassetvec, debtvec,  equityvec, cashvec, lagcapvec, FEIsimsMtx, ZsimsMtx, ZIsimsMtx,
                 asstsimsMtx, dividendsimsMtx, totKsimsMtx, NKratsimsMtx, cashsimsMtx, IRsimsMtx,
                 debtsimsMtx, NWsimsMtx, fracRepaidsimsMtx, outputsimsMtx,liqDecsimsMtx, agesimsMtx,
                 expensesimsMtx, liqDecisionMat, fracRepaidMat, bestIntRateFarm, bestCashFarm,
                 bestDividendFarm, bestKFarm, bestNKratFarm, bestDebtFarm, numSims)
