# Simon & Hjalte accidentally made the same function. 
# Is this a case for scrum?

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
	mean_fe = np.mean(TFP_FE, axis=0)

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
