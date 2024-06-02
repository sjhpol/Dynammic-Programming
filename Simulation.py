import numpy

from variables import *

from babyfarm18b import prodFnParms, numSims, ftNum, alpha1, gamma1, GetLocation, alpha2, gamma2, feNum, zNum2, lagcapNum, c_0, eqInject, getBaseIGoods, igshift, fixedcost, getBaseRevenues, delta, eGK


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
initAges = np.loadtxt(iopath + "initages.txt")
initYears = np.loadtxt(iopath + "inityrs.txt")
initTotAssets = np.loadtxt(iopath + "initta.txt")
initCapital = np.loadtxt(iopath + "initK.txt")
initDebt = np.loadtxt(iopath + "initdebt.txt")
zShks = np.loadtxt(iopath + "zshks.txt")
feShks = np.loadtxt(iopath + "feshks.txt")
FTypesims = np.loadtxt(iopath + "ftype_sim.txt")

recsize = (timespan + 1) * numSims

# Initialize arrays with NaN values
Zsims =  np.zeros(recsize)
ZIsims =  np.zeros(recsize)  # Index numbers, TFP shock
FEIsims =  np.zeros(recsize)  # Index numbers, fixed Effect TFP shock
asstsims =  np.zeros(recsize)  # Total Assets, beginning of period
debtsims =  np.zeros(recsize)  # Debt, beginning of period, pre-renegotiation
fracRepaidsims =  np.zeros(recsize)  # Fraction of outstanding debt repaid
liqDecsims =  np.zeros(recsize)  # Liquidation decisions
agesims =  np.zeros(recsize)  # Age of farm head
dividendsims =  np.zeros(recsize)  # Dividends/consumption
totKsims =  np.zeros(recsize)  # Capital Stock, beginning of period
NKratsims =  np.zeros(recsize)  # igoods/capital ratio
cashsims =  np.zeros(recsize)  # Cash/liquid assets
IRsims =  np.zeros(recsize)  # Contractual interest rates
NWsims =  np.zeros(recsize)  # Net worth for period, post-renegotiation
outputsims =  np.zeros(recsize)  # Output/revenues
expensesims =  np.zeros(recsize)  # Operating expenditures

# Reshape the arrays            # 
zShksMtx = zShks.reshape(timespan + 2, numSims)
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
      year0 = int(initYears[personInd]) - 1 # GAUSS indexing

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

      FEIsimsMtx[0][personInd] = feInd + 1 # GAUSS indexing                             # Float?
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

          #                                 #############   now move to t+1 states ###########

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
save_processed_data_txt(dividendsimsMtx, "divsS.txt", iopath)
save_processed_data_txt(totKsimsMtx, "totKS.txt", iopath)
save_processed_data_txt(NKratsimsMtx, "NKratos.txt", iopath)
save_processed_data_txt(cashsimsMtx, "cashS.txt", iopath)
save_processed_data_txt(IRsimsMtx, "intRateS.txt", iopath)
save_processed_data_txt(NWsimsMtx, "equityS.txt", iopath)
save_processed_data_txt(outputsimsMtx, "outputS.txt", iopath)
save_processed_data_txt(expensesimsMtx, "expenseS.txt", iopath)
