/*  BABYFARM18.C
  - Finds value fns and policy fns, and simulates them for the model(s) in Jones and Pratap
  - Written by John Jones, borrowing from code by Cristina De Nardi, Charles Doss, John Jones and Fang Yang
  - GAUSS-C I/O procedures written by K. Housinger (2003)
   - I/O instructions
    ~ Put the (GAUSS) *.fmt files in a subdirectory called \iofiles\
    ~ create a folder called \data
    ~ create a folder called \output
*/ 

// De magiske ord til Homebrew gcc Sonoma-compiler er, givet at du har en gcc-13 installation og ikke CLANG, 
// gcc-13 -fopenmp -o babyfarm18b babyfarm18b.c

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <limits.h>

#include <omp.h> // Open MP

// Disable warning messages 4996 
#pragma warning( once : 4996 )

#define NUM_THREADS 52   // PC has 8 threads, but save a thread for other work
#define switchMac 1     // 0: Sangeeta in Hunter, 1: John in Albany 
#define printOn 1        //'bool' for whether to print or not.  (will probably depend on job (whether estimating or not), MPI, etc.
#define errorPrintOn 1   //want separate tests for printing errors and printing other things.
#define PI 3.14159265
#define rounder 0.0001
#define ADDRESS_LEN 300
#define HSDIM 3          // Placeholder
#define tauBeq 0.0       // 0.15 Marginal tax rate on bequests per De Nardi
#define exBeq 600        // Bequest exemption level per De Nardi
#define minStepPct 0.05  // Minimum search interval in one direction, as fraction of control space
#define gridShrinkRate 0.03 // Rate at which interval shrinks, as fraction of control space
#define eRepayMin 1      // Limits excessive borrowing in final period of life
#define taxDim 7         // dim. of vector of marginal tax rates
#define eqInject 1       // allow dividends to be negative up to c_0

#define BASIC_HEADER_LEN 128 /* Length of a scalar header */
#define BYTE_POS 5           /* Int offset in header of byte order */
#define BIT_POS 6            /* Int offset in header of bit order */
#define TYPE_POS 15          /* Int offset in header of matrix type */
#define HEADER_LEN_POS 18    /* Int offset in header of header length */
#define M_POS 32             /* Int offset in header of value for m */
#define N_POS 33             /* Int offset in header of value for n */
#define SCALAR 0             /* Value of type that indicates a scalar */
#define MATRIX 2             /* Value of type that indicates a matrix */
#define BIT_SYSTEM 0         /* Value of bit order (0=backwards/i386) */
#define BYTE_SYSTEM 0        /* Value of byte order (0=backwards/i386) */

/*
**This is my GMatrix opaque datatype.  The gauss fmt format has implied 
**dimensionality, but is just a list of doubles.  I could rewrite the
**functions to use pointers-to-pointers to improve the interface with
**Eric's functions, but that would add to the complexity of my code.  I
**suggest writing an intermediate function that turns *data into **data
**to cope with 2-dimensional arrays.
*/ 
typedef struct {   /* this defines a matrix with given # of rows and */
  unsigned int m;  /* columns and gives the address of the first element */
  unsigned int n;
  double *data;} GMatrix;

/*-------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------Global PARAMETERS read in from GAUSS---------------------------------------------------*/

// Age and calendar time parameters
int bornage, retage, lifespan, firstyear, lastyear, timespan, numSims, noRenegotiation;
// Grid sizes
int zNum, zNum2, feNum, gkNum, gkNum2, assetNum, equityNum, totassetNum, debtNum, capitalNum, lagcapNum, NKratioNum, cashNum, ftNum;
double assetMin, assetMax, equityMin, equityMax, totassetMin, totassetMax, debtMin, debtMax,
       capitalMin, capitalMax, NKratioMin, NKratioMax, cashMin, cashMax;
// Preference parameters
double beta, nu, nu2, c_0, c_1, thetaB, chi, consFloor, cscale;
// Production parameters
double alpha1, gamma1, alpha2, gamma2, igshift, lambda, phi, SLfrac, zeta, delta, fixedcost, psi_inverse,
       r_riskfree, bigR, bigG, eGK, eRDG, rhoZ, stdZ, rhoFE, stdFE, rhoGK, stdGK;
  
int job; //read from GAUSS ( 1 => estimating and getting se's;2 => getting se's only; 3 => get graphs;  4 => experiments. )
int rank;   /* Index number of this node  */
int size;   /*  Number of nodes in cluster */

FILE *errorOutput; //initialized at the beginning of main and closed' at the end of main.  Use for all output.
//Log file; for knowing what you've done.
FILE *logOutput;
FILE *parmValues;

/* Income tax structure, from French's code
   - taxBrk gives tax brackets
   - taxMar gives marginal tax rates  
*/
double taxBrk[taxDim-1] = {7490, 48170, 89150, 112570, 177630, 342120}; // JBJ:  Switched to 2005 dollars, 11/12/12
double taxMar[taxDim] = {0.0765, 0.2616, 0.4119, 0.3499, 0.3834, 0.4360, 0.4761}; 
double incomeBrk[taxDim-1];   // income after tax at brackets 

/* Strings of directory names */
char rootdir[ADDRESS_LEN]; 
char outputdir[ADDRESS_LEN]; 
char datadir[ADDRESS_LEN]; 

struct IndexInfo
{ int Ind1;
double weight;
};

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/* functions used in main*/

//  GAUSS-C++ I/O programs written by K. Housinger 
unsigned char * gread(unsigned char *inbuf, int bytes, int byte_reverse, int bit_reverse);
GMatrix gau5read(char *fmt);
GMatrix zeroGMat(unsigned int recsize);
void gau5write(char *fmt, GMatrix mat); /* reads vector from hard drive*/

//  Converts vectors to matrices and vice-versa
double **SetUp2DMat(double *dataVec, int numRows, int numCols);
double ***SetUp3DMat(double *dataVec, int numDim1, int numRows, int numCols);
double ****SetUp4DMat(double *dataVec, int numDim1, int numDim2, int numRows, int numCols);
double **ZeroMat2D(int numRows, int numCols);
double ***ZeroMat3D(int numDim1, int numRows, int numCols);
double ****ZeroMat4D(int numDim1, int numDim2, int numRows, int numCols);
double *****ZeroMat5D(int numDim1, int numDim2, int numDim3, int numRows, int numCols);
double ******ZeroMat6D(int numDim1, int numDim2, int numDim3, int numDim4, int numRows, int numCols);
double *******ZeroMat7D(int numDim1, int numDim2, int numDim3, int numDim4, int numDim5, int numRows, int numCols);
void FreeMat2D(double **dataMat, int numRows, int numCols);
void FreeMat3D(double ***dataMat, int numDim1, int numRows, int numCols);
void FreeMat4D(double ****dataMat, int numDim1, int numDim2, int numRows, int numCols);
void FreeMat5D(double *****dataMat, int numDim1, int numDim2, int numDim3, int numRows, int numCols);
void FreeMat6D(double ******dataMat, int numDim1, int numDim2, int numDim3, int numDim4, 
               int numRows, int numCols);
void FreeMat7D(double *******dataMat, int numDim1, int numDim2, int numDim3, int numDim4, 
               int numDim5, int numRows, int numCols);
int **ZeroMat2DI(int numRows, int numCols);
int ***ZeroMat3DI(int numDim1, int numRows, int numCols);
int ****ZeroMat4DI(int numDim1, int numDim2, int numRows, int numCols);
int *****ZeroMat5DI(int numDim1, int numDim2, int numDim3, int numRows, int numCols);
int ******ZeroMat6DI(int numDim1, int numDim2, int numDim3, int numDim4, int numRows, int numCols);
int *******ZeroMat7DI(int numDim1, int numDim2, int numDim3, int numDim4, int numDim5, int numRows, int numCols);
void FreeMat2DI(int **dataMat, int numRows, int numCols);
void FreeMat3DI(int ***dataMat, int numDim1, int numRows, int numCols);
void FreeMat4DI(int ****dataMat, int numDim1, int numDim2, int numRows, int numCols);
void FreeMat5DI(int *****dataMat, int numDim1, int numDim2, int numDim3, int numRows, int numCols);
void FreeMat6DI(int ******dataMat, int numDim1, int numDim2, int numDim3, int numDim4, 
                int numRows, int numCols);
void FreeMat7DI(int *******dataMat, int numDim1, int numDim2, int numDim3, int numDim4, 
                int numDim5, int numRows, int numCols);

//   Update global parameters with values passed in from GAUSS
int globderef(double *agevecPtr, double *sizevecPtr, double *prefvecPtr, 
              double *finvecPtr, double *initWage);
void prodFnParms(double alpha0, double gamma0, double *alpha, double *gamma, 
                 double *ag2, double *ag3, double *gag, double *agag);
double **getMarkovChain(double *chainInfoPtr, int numStates, double *rho, double *std, 
                        double *values, double *piInvarV);
double **GetCDFmtx(int nRows, int nCols, double **transmtx); 
double *GetCDFvec(int nCols, double *probvec) ;
void IncomeAtBrk(double taxBrk[], double taxMar[], double incomeBrk[]);  // after-tax income at bracket points 
double AfterTaxIncome(double y);

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

int Locate(double *Xarray, double x, int DIM);
int LocateClosest(double *Xarray, double x, int DIM);
struct IndexInfo GetLocation(double *xP, double x, int DIM);

double XtimesBeta(double *coeffs, double age, int sexInd, double PInc, int healthStat);
double GetTransProbs(double **hsCoeffs, double age, int sexInd, 
                     double PInc, int healthStat, double HSProbs[]);
double LogitSQRT(double x);
double Logit(double x);

void getliqIndex(double *assetvec, double *totassetvec, double *lagcapvec, 
                 double *debtvec, double ***postliqAssets, double ***postliqNetWorth, 
                 double ***postliqNWIndex);
void getNPtotAssets(int iTotKmin, int iTotKmax, double *capitalvec, double *NKratiovec, 
                    double *cashvec, double *feValues, double *zValues, double *totassetvec, 
                    double ******NPtotassetWeight, int ******NPtotassetIndex, 
                    int *****goodGridPoint);
double getExpectation(double ****RandVar, int *NPTAIndex, double *NPTAWeight,
                      int iLagK, double iLagKwgt, int iDebt, double *zProbs);
double intrplte7D(double *******decruleMat, int ageInd, int ftInd, int feInd, int zInd2, int lkInd2, int taInd, int dInd,
                  double dWgt, double taWgt, double feWgt, double zWgt, double lkWgt); 
double intrplte6D(double ******decruleMat, int ageInd, int ftInd, int feInd, int zInd2, int lkInd2, int eqInd, 
                  double eqWgt, double feWgt, double zWgt, double lkWgt);                                    
double getUtility(double cons);
double NetBequest(double amountBequestedP);
double UBeqSingle(double netBequestP);
void GetUtilityBeq(double bequestUM[], double astate[]);
double getBaseRevenues(double capital, double igoods, double gamma, double ag2);
double getbaseIGoods(double capital, double eTFP, double gag, double agag, double ag3);
void getclosestLK(int numPTrue, int numPTarget, double *truevec, double *targetvec, 
                  int *CLKindvec, double *CLKwgtvec);
int *getAssignmentVec(int numPoints, int numNodes);

void GetRulesWorker(int iAssetsmin, int iAssetsmax, double *assetvec, double *wageProfile, 
                    double **valfuncWork, double **bestCWork, double **bestNPIWork);
void getFinalLiq(int iTotAssetmin, int iTotAssetmax, double *totassetvec, 
                 double *lagcapvec, double *debtvec, double *assetvec,  
                 double ***retNWIndex,double *******liqDecisionMat, 
                 double *******valfuncMat, double *******fracRepaidMat);
void getliqDecision(double *totassetvec, double *debtvec, double *equityvec,  
                    double ***postliqAssets, double ***postliqNWIndex, double **valfuncWork,  
                    double ******valfuncFarm, double *******liqDecisionMat, 
                    double *******valfuncMat, double *******fracRepaidMat,
                    int iTotAssetMin, int iTotAssetMax, int ageInd);
void getOperatingDec(int jEquityMin, int jEquityMax, int *eqAssignvec, int ageInd, 
                     double *equityvec, double *capitalvec, double *lagcapvec, double *debtvec, 
                     double *NKratiovec, double *cashvec, double **zTransmtx, int *CLKindvec, 
                     double *CLKwgtvec,
                     int *****goodGridPoint, double ******NPtotassetWeight, int ******NPtotassetIndex, 
                     double *******valfuncMat, double *******fracRepaidVec, 
                     double ******valfuncFarm, double ******bestIntRateFarm, int ******bestKIndexFarm, 
                     int ******bestNKratIndexFarm, int ******bestCashIndexFarm, 
                     double ******bestDividendFarm, int ******bestDebtIndexFarm, 
                     double *******bestNPTAWeightFarm, int*******bestNPTAIndexFarm, 
                     double ******bestKFarm, double ******bestNKratFarm, double ******bestDebtFarm, 
                     double ******bestCashFarm);
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

// Hjaltes nye .txt-funktioner.
GMatrix readDoubleList(const char* filename);
void writeGMatrixToFile(const char* filename, const GMatrix* mat);

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/* main program, note that everything before this is global */
int main(int argc, char *argv[])
{
   GMatrix GaussJobPtr, agevecPtr, sizevecPtr, prefvecPtr, finvecPtr, zvecPtr, gkvecPtr, 
           fevecPtr, assetPtr, equityPtr, totassetPtr, debtPtr, capitalPtr, lagcapPtr, NKratioPtr, 
           cashPtr, wageprofilePtr, valfFuncWPtr, bestCWPtr, bestNPIWPtr, initAgesPtr, initYearsPtr,
           initTotAssetsPtr, initCapitalPtr, initDebtPtr, zShksPtr, feShksPtr, ZsimsPtr, ZIsimsPtr, 
           FEIsimsPtr, asstsimsPtr, dividendsimsPtr, totKsimsPtr, NKratsimsPtr, cashsimsPtr, 
           IRsimsPtr, debtsimsPtr, NWsimsPtr, fracRepaidsimsPtr, outputsimsPtr, liqDecsimsPtr, 
           agesimsPtr, expensesimsPtr, FTypesimsPtr;

   double *assetvec, *equityvec, *totassetvec, *debtvec, *capitalvec, *lagcapvec, *CLKwgtvec, 
          *NKratiovec, *cashvec, *bequestUM, *zValues, *zInvarDist, *zInvarCDF, *feValues, 
          *feProbs, *gkValues, *gkInvarDist, *gkInvarCDF;
   double **zTransmtx, **zTransCDFmtx, **gkTransmtx, **gkTransCDFmtx, **feProbs2;
// Worker:  state space = Age x Assets
   double **valfuncWork, **bestCWork, **bestNPIWork;
// Mapping from Lagged Capital x TotAssets x Debt to post-Liquidation Net Worth/Assets
   double ***postliqAssets, ***postliqNetWorth, ***postliqNWIndex, ***retNWIndex;
// Operating Farm:  state space = Prodn Type x Age x FE x TFP x Lagged Capital x Net Worth
   double ******valfuncFarm, ******bestIntRateFarm, ******bestDividendFarm, ******bestKFarm, ******bestNKratFarm, 
          ******bestDebtFarm, ******bestCashFarm;
   int ******bestKIndexFarm, ******bestNKratIndexFarm, ******bestCashIndexFarm, ******bestDebtIndexFarm;
   double *******bestNPTAWeightFarm;  // These map to next period total asset grid
   int *******bestNPTAIndexFarm;      // Last dimension is realized time-t+1 TFP
// Prior to continued operation decision
// State space = Prodn Type x Age x FE x TFP x Lagged Capital x TotAssets x Debt 
   double *******valfuncMat, *******fracRepaidMat, *******liqDecisionMat;

   int *****goodGridPoint; // Decision grid:  TotCapital x FE x NKratio x CashExpenses
   double ******NPtotassetWeight; // This is for decision grid, mapping transitions
   int ******NPtotassetIndex;     // Last dimension is realized time-t+1 TFP
   int *CLKindvec, *equityAssignments;

// Simulation Results
   double **zShksMtx, **ZsimsMtx, **ZIsimsMtx, **FEIsimsMtx, **asstsimsMtx, **dividendsimsMtx, **totKsimsMtx, 
          **NKratsimsMtx, **cashsimsMtx, **IRsimsMtx, **debtsimsMtx, **NWsimsMtx, **fracRepaidsimsMtx,
          **outputsimsMtx, **liqDecsimsMtx, **agesimsMtx, **expensesimsMtx;
   int job, gotderef, recsize, tInd;  //yearInd
   char fullpath[ADDRESS_LEN];

   clock_t start, end;  /* recode time */

   int simspermachine, iAssetsmin, iAssetsmax, iSimmin, iSimmax, statespermachine, assetspermachine,
       iTotKmin, iTotKmax, iTotAssetmin, iTotAssetmax, iEquitymin, iEquitymax;
   double spm2;

   FILE *timeStamp;  // an ID for each run and has length of run.

   // The switch is defined in the top

   if (switchMac==1)  // Hjalte
   {
      strcpy(rootdir,"/Users/hjaltewallin/Code/DP-MESTERNE/Dynammic-Programming/iofiles/");
      strcpy(outputdir, "/Users/hjaltewallin/Code/DP-MESTERNE/Dynammic-Programming/output/"); 
      strcpy(datadir, "/Users/hjaltewallin/Code/DP-MESTERNE/Dynammic-Programming/data/"); 
   }
   else if (switchMac==0) 
   {
      strcpy(rootdir, "c:\\users\\sangeeta\\dropbox\\farms\\estimation_fake\\iofiles\\");
      strcpy(outputdir, "c:\\users\\sangeeta\\dropbox\\farms\\estimation_fake\\output\\");
      strcpy(datadir, "c:\\users\\sangeeta\\dropbox\\farms\\estimation_fake\\data\\");
   }
  
   GaussJobPtr = readDoubleList(strcat(strcpy(fullpath,rootdir),"job.txt"));
   job   = (int) floor(rounder+GaussJobPtr.data[0]); // job is global
   
   start = clock();
   rank  = 0; 
   size  = 1;

// Get number of processors.
   omp_set_num_threads(NUM_THREADS);
   #pragma omp parallel 
   {   size = omp_get_num_threads(); }
   printf("Number of threads = %d\n", size);

   timeStamp = fopen(strcat(strcpy(fullpath,outputdir), "timeStamp.txt"), "w");
   errorOutput=fopen(strcat(strcpy(fullpath,outputdir), "erroroutput.txt"), "w");
 
   if (printOn>0)
      fprintf(timeStamp, "ID: %f\n", (float)start); //ID==start is the time at which the program was run.

// Read in parameter vectors from the *txt files (read into memory as arrays)
   agevecPtr      = readDoubleList(strcat(strcpy(fullpath,rootdir),"agevec.txt")); 
   sizevecPtr     = readDoubleList(strcat(strcpy(fullpath,rootdir),"sizevec.txt")); 
   prefvecPtr     = readDoubleList(strcat(strcpy(fullpath,rootdir),"pref_parms.txt"));
   finvecPtr      = readDoubleList(strcat(strcpy(fullpath,rootdir),"fin_parms.txt"));  
   assetPtr       = readDoubleList(strcat(strcpy(fullpath,rootdir),"Astate.txt"));
   equityPtr      = readDoubleList(strcat(strcpy(fullpath,rootdir),"Estate.txt"));
   totassetPtr    = readDoubleList(strcat(strcpy(fullpath,rootdir),"TAstate.txt"));
   debtPtr        = readDoubleList(strcat(strcpy(fullpath,rootdir),"Bstate.txt"));
   capitalPtr     = readDoubleList(strcat(strcpy(fullpath, rootdir),"Kstate.txt"));
   lagcapPtr      = readDoubleList(strcat(strcpy(fullpath,rootdir),"lagKstate.txt"));
   NKratioPtr     = readDoubleList(strcat(strcpy(fullpath,rootdir),"NKstate.txt"));
   cashPtr        = readDoubleList(strcat(strcpy(fullpath,rootdir),"Cstate.txt"));
   zvecPtr        = readDoubleList(strcat(strcpy(fullpath,rootdir),"zvec.txt"));
   fevecPtr       = readDoubleList(strcat(strcpy(fullpath,rootdir),"fevec.txt"));
   gkvecPtr       = readDoubleList(strcat(strcpy(fullpath,rootdir),"gkvec.txt"));
   wageprofilePtr = readDoubleList(strcat(strcpy(fullpath,rootdir),"wprof.txt"));

   assetvec       = assetPtr.data;
   equityvec      = equityPtr.data;
   totassetvec    = totassetPtr.data;
   debtvec        = debtPtr.data;
   capitalvec     = capitalPtr.data;
   lagcapvec      = lagcapPtr.data;
   NKratiovec     = NKratioPtr.data;
   cashvec        = cashPtr.data;
   assetNum       = assetPtr.m;
   equityNum      = equityPtr.m;
   totassetNum    = totassetPtr.m;
   debtNum        = debtPtr.m;
   capitalNum     = capitalPtr.m;
   NKratioNum     = NKratioPtr.m;
   cashNum        = cashPtr.m;

// Initialize parameters by assigning array elements to appropriate parameters
   gotderef   = globderef(agevecPtr.data, sizevecPtr.data, prefvecPtr.data, 
                          finvecPtr.data, wageprofilePtr.data);

// Lagged capital grid differs from capital decision grid.  Find mapping from one to the other
   lagcapNum  = lagcapPtr.m;
   CLKindvec  = (int *)calloc(capitalNum,sizeof(int));
   CLKwgtvec  = (double *)calloc(capitalNum,sizeof(double));
   getclosestLK(capitalNum, lagcapNum, capitalvec, lagcapvec, CLKindvec, CLKwgtvec);
   if (phi==0) { lagcapNum = 1; }

   zNum       = (int) floor(rounder+zvecPtr.data[0]);  // TFP shock
   zValues    = (double *)calloc(zNum,sizeof(double));
   zInvarDist = (double *)calloc(zNum,sizeof(double));
   zTransmtx  = getMarkovChain(zvecPtr.data, zNum, &rhoZ, &stdZ, zValues, zInvarDist);
   zNum2      = zNum; // dimension for state vector
   if (rhoZ==0) { zNum2 = 1; }
   zTransCDFmtx = GetCDFmtx(zNum, zNum, zTransmtx); 
   zInvarCDF  = GetCDFvec(zNum, zInvarDist); 

   feNum      = (int)floor(rounder + fevecPtr.data[0]);  // TFP shock
   feValues   = (double *)calloc(feNum, sizeof(double));
   feProbs    = (double *)calloc(feNum, sizeof(double));
   feProbs2   = getMarkovChain(fevecPtr.data, feNum, &rhoFE, &stdFE, feValues, feProbs); // Lots of placeholders here

   gkNum       = (int) floor(rounder+gkvecPtr.data[0]); // Capital gains shock, conditionally i.i.d.
   gkValues    = (double *)calloc(gkNum,sizeof(double));
   gkInvarDist = (double *)calloc(gkNum,sizeof(double));
   gkTransmtx  = getMarkovChain(gkvecPtr.data, gkNum, &rhoGK, &stdGK, gkValues, gkInvarDist);
   gkNum2      = gkNum; // dimension for state vector
   if (rhoGK==0) { gkNum2 = 1; }
   gkTransCDFmtx = GetCDFmtx(gkNum, gkNum, gkTransmtx); 
   gkInvarCDF  = GetCDFvec(gkNum, gkInvarDist); 

// Initialize matrices
   bequestUM = (double *)calloc(assetNum,sizeof(double));
   GetUtilityBeq(bequestUM,assetvec); /* Find utility from bequest matrix */
   IncomeAtBrk(taxBrk, taxMar, incomeBrk); /* after-tax income at bracket points  */

   recsize         = (lifespan+1)*assetNum;
   valfFuncWPtr    = zeroGMat(recsize);
   recsize         = lifespan*assetNum;
   bestCWPtr       = zeroGMat(recsize);  // optimal consumption choices
   bestNPIWPtr     = zeroGMat(recsize);  // index number of optimal time-t+1 assets

   valfuncWork     = SetUp2DMat(valfFuncWPtr.data,lifespan+1,assetNum);
   valfuncWork[lifespan] = bequestUM;
   bestCWork       = SetUp2DMat(bestCWPtr.data,lifespan,assetNum);
   bestNPIWork     = SetUp2DMat(bestNPIWPtr.data,lifespan,assetNum);

   spm2 = ((double) assetNum)/ ((double) size);
   statespermachine = (int) ceil(spm2);
   spm2 = ((double) numSims)/ ((double) size);
   simspermachine = (int) ceil(spm2);

   #pragma omp parallel private(iAssetsmin,iAssetsmax,rank)
   {
      rank = omp_get_thread_num();
      iAssetsmin = rank*statespermachine;
      iAssetsmax = (rank+1)*statespermachine;
      if (iAssetsmax > assetNum) { iAssetsmax = assetNum; }
      GetRulesWorker(iAssetsmin,iAssetsmax,assetvec,wageprofilePtr.data, 
                     valfuncWork, bestCWork, bestNPIWork);
   }

   printf("Finished with decision rules for workers (rank==%d).\n", rank);

   if (printOn==2)
   {
      writeGMatrixToFile(strcat(strcpy(fullpath,rootdir),"vfWork.fmt"), &valfFuncWPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath,rootdir),"bestCWork.fmt"), &bestCWPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath,rootdir),"bestNPIWork.fmt"), &bestNPIWPtr);
      end = clock();
      printf("Worker's problem ends in %5d minutes %5d seconds \n ",(end-start)/CLOCKS_PER_SEC/60,
           ((end-start)/CLOCKS_PER_SEC)%60);
      fprintf(timeStamp, "Worker's problem ends in %5d minutes %5d seconds \n ",(end-start)/CLOCKS_PER_SEC/60,
            ((end-start)/CLOCKS_PER_SEC)%60);
      start = clock();
   }
   fclose(timeStamp); //didn't open unless printOn>0.

// Calculate mapping from TotAssets x Lagged Capital x Debt to post-Liquidation Net Worth/Assets
// These are age-invariant, financial calculations

   postliqAssets    = ZeroMat3D(lagcapNum, totassetNum, debtNum);  // Assets after liquidation
   postliqNetWorth  = ZeroMat3D(lagcapNum, totassetNum, debtNum);  // Net Worth after liquidation 
   postliqNWIndex   = ZeroMat3D(lagcapNum, totassetNum, debtNum);  // Index number of point asset grid closest to net worth
   retNWIndex       = ZeroMat3D(lagcapNum, totassetNum, debtNum);  // Index number of point asset grid closest to post-retirement net worth

   getliqIndex(assetvec, totassetvec, lagcapvec, debtvec, postliqAssets, postliqNetWorth, postliqNWIndex);

// Calculate mapping from farmer's decision grid (total Capital x FE x NKratio x Cash expenditures ) to future (total) assets

   NPtotassetWeight = ZeroMat6D(capitalNum, ftNum, feNum, NKratioNum, cashNum, zNum);
   NPtotassetIndex  = ZeroMat6DI(capitalNum, ftNum, feNum, NKratioNum, cashNum, zNum);
   goodGridPoint    = ZeroMat5DI(capitalNum, ftNum, feNum, NKratioNum, cashNum);

   spm2 = ((double) capitalNum)/ ((double) size);
   assetspermachine = (int) ceil(spm2);

   #pragma omp parallel private(iTotKmin,iTotKmax,rank)
   {
      rank = omp_get_thread_num();
      iTotKmin = rank*assetspermachine;
      iTotKmax = (rank+1)*assetspermachine;
      if (iTotKmax > capitalNum) { iTotKmax = capitalNum; }
      getNPtotAssets(iTotKmin, iTotKmax, capitalvec, NKratiovec, cashvec, feValues, zValues, 
                     totassetvec, NPtotassetWeight, NPtotassetIndex, goodGridPoint);
   }               
  
   liqDecisionMat     = ZeroMat7D(lifespan+1, ftNum, feNum, zNum2, lagcapNum, totassetNum, debtNum);  // Occupation choice
   valfuncMat         = ZeroMat7D(lifespan+1, ftNum, feNum, zNum2, lagcapNum, totassetNum, debtNum);  // Value Function allowing for occupation decision
   fracRepaidMat      = ZeroMat7D(lifespan+1, ftNum, feNum, zNum2, lagcapNum, totassetNum, debtNum);  // Fraction of debt repaid
   valfuncFarm        = ZeroMat6D(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum); // Value function for Operating farm
   bestIntRateFarm    = ZeroMat6D(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum); 
   bestKIndexFarm     = ZeroMat6DI(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum);
   bestKFarm          = ZeroMat6D(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum);
   bestNKratIndexFarm = ZeroMat6DI(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum); 
   bestNKratFarm      = ZeroMat6D(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum);
   bestCashIndexFarm  = ZeroMat6DI(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum); 
   bestCashFarm       = ZeroMat6D(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum); 
   bestDividendFarm   = ZeroMat6D(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum); 
   bestDebtIndexFarm  = ZeroMat6DI(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum); 
   bestDebtFarm       = ZeroMat6D(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum);
   bestNPTAWeightFarm = ZeroMat7D(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum, zNum);
   bestNPTAIndexFarm  = ZeroMat7DI(lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum, zNum);
     
   spm2 = ((double) totassetNum)/ ((double) size);
   assetspermachine = (int) ceil(spm2);
   spm2 = ((double) equityNum)/ ((double) size);
   statespermachine = (int) ceil(spm2);

   #pragma omp parallel private(iTotAssetmin,iTotAssetmax,rank)
   {
      rank = omp_get_thread_num();
      iTotAssetmin = rank*assetspermachine;
      iTotAssetmax = (rank+1)*assetspermachine;
      if (iTotAssetmax > totassetNum) { iTotAssetmax = totassetNum; }
     
      getFinalLiq(iTotAssetmin, iTotAssetmax, totassetvec, lagcapvec, debtvec, assetvec,
                  retNWIndex, liqDecisionMat, valfuncMat, fracRepaidMat);
   }
     
   end = clock();
   printf("Setup calcs end in %5d minutes %5d seconds \n ",(end-start)/CLOCKS_PER_SEC/60,
          ((end-start)/CLOCKS_PER_SEC)%60);
 //  fprintf(timeStamp, "Setup calcs end in %5d minutes %5d seconds \n ",(end-start)/CLOCKS_PER_SEC/60,
 //          ((end-start)/CLOCKS_PER_SEC)%60);
   start = clock();

   equityAssignments = getAssignmentVec(equityNum, size);
   
   for (tInd = (lifespan-1); tInd>= 0; tInd--)
   {  
      printf("tInd %5d\n", tInd);
         
      #pragma omp parallel private(iEquitymin,iEquitymax,rank)
      {
         rank = omp_get_thread_num();
         iEquitymin = rank*statespermachine;
         iEquitymax = (rank+1)*statespermachine;
         if (iEquitymax > equityNum) { iEquitymax = equityNum; }
      
         getOperatingDec(iEquitymin, iEquitymax, equityAssignments, tInd, equityvec, capitalvec, 
                         lagcapvec, debtvec, NKratiovec, cashvec, zTransmtx, CLKindvec, CLKwgtvec, 
                         goodGridPoint, NPtotassetWeight, NPtotassetIndex, valfuncMat, fracRepaidMat, 
                         valfuncFarm, bestIntRateFarm, bestKIndexFarm, bestNKratIndexFarm, 
                         bestCashIndexFarm, bestDividendFarm ,bestDebtIndexFarm, bestNPTAWeightFarm, 
                         bestNPTAIndexFarm, bestKFarm, bestNKratFarm, bestDebtFarm, bestCashFarm);
      }
      
      if (printOn>0)
      {
         end = clock();
         printf("Finished period %5d operating decisions in %5d minutes %5d seconds \n ",tInd, (end-start)/CLOCKS_PER_SEC/60,
                ((end-start)/CLOCKS_PER_SEC)%60);
//         fprintf(timeStamp, "Finished period %5d operating decisions in %5d minutes %5d seconds \n ",tInd, (end-start)/CLOCKS_PER_SEC/60,
//                 ((end-start)/CLOCKS_PER_SEC)%60);
         start = clock();
      }
      
      #pragma omp parallel private(iTotAssetmin,iTotAssetmax,rank)
      {
         rank = omp_get_thread_num();
         iTotAssetmin = rank*assetspermachine;
         iTotAssetmax = (rank+1)*assetspermachine;
         if (iTotAssetmax > totassetNum) { iTotAssetmax = totassetNum; }

         getliqDecision(totassetvec, debtvec, equityvec, postliqAssets, postliqNWIndex,  
                        valfuncWork, valfuncFarm, liqDecisionMat, valfuncMat, fracRepaidMat, 
                        iTotAssetmin, iTotAssetmax, tInd);
      }
     
      if (printOn>0)
      {
         end = clock();
         printf("Finished period %5d occupation decisions in %5d minutes %5d seconds \n ",tInd, (end-start)/CLOCKS_PER_SEC/60,
                ((end-start)/CLOCKS_PER_SEC)%60);
//         fprintf(timeStamp, "Finished period %5d occupation decisions in %5d minutes %5d seconds \n ",tInd, (end-start)/CLOCKS_PER_SEC/60,
//                ((end-start)/CLOCKS_PER_SEC)%60);
         fclose(timeStamp);
         timeStamp = fopen(strcat(strcpy(fullpath,outputdir), "timeStamp.txt"), "a+");
         start = clock();
      }
   }

   if (printOn==2)
   {
      printf("Printing Decision Rules.\n");
      WriteFunctions(valfuncWork, bestCWork, bestNPIWork, valfuncFarm, bestIntRateFarm, 
                     bestCashFarm, bestKFarm, bestNKratFarm, bestDividendFarm, bestDebtFarm, 
                     liqDecisionMat, valfuncMat, fracRepaidMat, assetvec, 
                     equityvec, lagcapvec, debtvec, totassetvec, feValues);
      end = clock();
      printf("Finished printing decision rules in %5d minutes %5d seconds \n ", (end-start)/CLOCKS_PER_SEC/60,
             ((end-start)/CLOCKS_PER_SEC)%60);
      fprintf(timeStamp, "Finished printing decision rules in %5d minutes %5d seconds \n ",(end-start)/CLOCKS_PER_SEC/60,
             ((end-start)/CLOCKS_PER_SEC)%60);
      fclose(timeStamp);
      timeStamp = fopen(strcat(strcpy(fullpath,outputdir), "timeStamp.txt"), "a+");
      start = clock();
   }

// Free up some memory
   FreeMat6D(NPtotassetWeight, capitalNum, ftNum, feNum, NKratioNum, cashNum, zNum);
   FreeMat6DI(NPtotassetIndex, capitalNum, ftNum, feNum, NKratioNum, cashNum, zNum);
   FreeMat5DI(goodGridPoint, capitalNum, ftNum, feNum, NKratioNum, cashNum);
   FreeMat7D(valfuncMat, lifespan+1, ftNum, feNum, zNum2, lagcapNum, totassetNum, debtNum);
   FreeMat6D(valfuncFarm, lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum);
   FreeMat6DI(bestKIndexFarm, lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum);
   FreeMat6DI(bestNKratIndexFarm, lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum);
   FreeMat6DI(bestCashIndexFarm, lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum);
   FreeMat6DI(bestDebtIndexFarm, lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum);
   FreeMat7D(bestNPTAWeightFarm, lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum, zNum);
   FreeMat7DI(bestNPTAIndexFarm, lifespan, ftNum, feNum, zNum2, lagcapNum, equityNum, zNum);

// On to the simulations

   initAgesPtr       = readDoubleList(strcat(strcpy(fullpath,rootdir),"initages.txt")); 
   initYearsPtr      = readDoubleList(strcat(strcpy(fullpath, rootdir),"inityrs.txt"));
   initTotAssetsPtr  = readDoubleList(strcat(strcpy(fullpath,rootdir),"initta.txt"));
   initCapitalPtr    = readDoubleList(strcat(strcpy(fullpath, rootdir), "initK.txt"));
   initDebtPtr       = readDoubleList(strcat(strcpy(fullpath,rootdir),"initdebt.txt"));
   zShksPtr          = readDoubleList(strcat(strcpy(fullpath,rootdir),"zshks.txt")); 
   feShksPtr         = readDoubleList(strcat(strcpy(fullpath, rootdir), "feshks.txt"));
   FTypesimsPtr      = readDoubleList(strcat(strcpy(fullpath, rootdir), "ftype_sim.txt"));

   recsize           = (timespan+1)*numSims;
   ZsimsPtr          = zeroGMat(recsize);  // Simulated values, transitory TFP shock
   ZIsimsPtr         = zeroGMat(recsize);  // Index numbers, TFP shock
   FEIsimsPtr        = zeroGMat(recsize);  // Index numbers, fixed Effect TFP shock  
   asstsimsPtr       = zeroGMat(recsize);  // Total Assets, beginning of period
   debtsimsPtr       = zeroGMat(recsize);  // Debt, beginning of period, pre-renegotiation
   fracRepaidsimsPtr = zeroGMat(recsize);  // Fraction of outstanding debt repaid
   liqDecsimsPtr     = zeroGMat(recsize);  // Liquidation decisions
   agesimsPtr        = zeroGMat(recsize);  // Age of farm head
   dividendsimsPtr   = zeroGMat(recsize);  // Dividends/consumption 
   totKsimsPtr       = zeroGMat(recsize);  // Capital Stock, beginning of period
   NKratsimsPtr      = zeroGMat(recsize);  // igoods/capital ratio
   cashsimsPtr       = zeroGMat(recsize);  // Cash/liquid assets 
   IRsimsPtr         = zeroGMat(recsize);  // Contractual interest rates
   NWsimsPtr         = zeroGMat(recsize);  // Net worth for period, post-renegotiation
   outputsimsPtr     = zeroGMat(recsize);  // Output/revenues
   expensesimsPtr    = zeroGMat(recsize);  // Operating expenditures

   zShksMtx          = SetUp2DMat(zShksPtr.data,timespan+2,numSims); // Note extra dummy year
   ZsimsMtx          = SetUp2DMat(ZsimsPtr.data,timespan+1,numSims); 
   ZIsimsMtx         = SetUp2DMat(ZIsimsPtr.data,timespan+1,numSims);
   FEIsimsMtx        = SetUp2DMat(FEIsimsPtr.data,timespan+1,numSims);
   asstsimsMtx       = SetUp2DMat(asstsimsPtr.data,timespan+1,numSims);
   debtsimsMtx       = SetUp2DMat(debtsimsPtr.data,timespan+1,numSims);
   fracRepaidsimsMtx = SetUp2DMat(fracRepaidsimsPtr.data,timespan+1,numSims);
   liqDecsimsMtx     = SetUp2DMat(liqDecsimsPtr.data,timespan+1,numSims);
   agesimsMtx        = SetUp2DMat(agesimsPtr.data,timespan+1,numSims);
   dividendsimsMtx   = SetUp2DMat(dividendsimsPtr.data,timespan+1,numSims);
   totKsimsMtx       = SetUp2DMat(totKsimsPtr.data,timespan+1,numSims);
   NKratsimsMtx      = SetUp2DMat(NKratsimsPtr.data,timespan+1,numSims);
   cashsimsMtx       = SetUp2DMat(cashsimsPtr.data,timespan+1,numSims);
   IRsimsMtx         = SetUp2DMat(IRsimsPtr.data,timespan+1,numSims);
   NWsimsMtx         = SetUp2DMat(NWsimsPtr.data,timespan+1,numSims);
   expensesimsMtx    = SetUp2DMat(expensesimsPtr.data,timespan+1,numSims);
   outputsimsMtx     = SetUp2DMat(outputsimsPtr.data,timespan+1,numSims);

   #pragma omp parallel private(iSimmin,iSimmax,rank)
   {
      rank = omp_get_thread_num();
      iSimmin = rank*simspermachine;
      iSimmax = (rank+1)*simspermachine;
      if (iSimmax > numSims) { iSimmax = numSims; }
//      iSimmin = 0;
//      iSimmax = numSims;

      simulation(initAgesPtr.data, initYearsPtr.data, initCapitalPtr.data, initTotAssetsPtr.data, 
                 initDebtPtr.data, FTypesimsPtr.data, feShksPtr.data, feValues, zShksMtx, zValues, 
                 totassetvec, debtvec,  equityvec, cashvec, lagcapvec, FEIsimsMtx, ZsimsMtx, ZIsimsMtx, 
                 asstsimsMtx, dividendsimsMtx, totKsimsMtx, NKratsimsMtx, cashsimsMtx, IRsimsMtx, 
                 debtsimsMtx, NWsimsMtx, fracRepaidsimsMtx, outputsimsMtx,liqDecsimsMtx, agesimsMtx, 
                 expensesimsMtx, liqDecisionMat, fracRepaidMat, bestIntRateFarm, bestCashFarm, 
                 bestDividendFarm, bestKFarm, bestNKratFarm, bestDebtFarm, iSimmin, iSimmax);
   }

   if (printOn>0)
   {
      printf("Printing Simulations.\n");
      if (printOn == 2)
      {
         WriteSims(FEIsimsMtx, ZsimsMtx, ZIsimsMtx, asstsimsMtx, dividendsimsMtx, totKsimsMtx,
                   NKratsimsMtx, cashsimsMtx, IRsimsMtx, debtsimsMtx, NWsimsMtx, fracRepaidsimsMtx,
                   outputsimsMtx, liqDecsimsMtx, agesimsMtx, expensesimsMtx);
      }
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "FEindxS.txt"), &FEIsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "ZValsS.txt"), &ZsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "ZindxS.txt"), &ZIsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "assetsS.txt"), &asstsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "debtS.txt"), &debtsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "fracRPS.txt"), &fracRepaidsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "liqDecS.txt"), &liqDecsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "ageS.txt"), &agesimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "divsS.txt"), &dividendsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "totKS.txt"), &totKsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "NKratioS.txt"), &NKratsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "cashS.txt"), &cashsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "intRateS.txt"), &IRsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "equityS.txt"), &NWsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "outputS.txt"), &outputsimsPtr);
      writeGMatrixToFile(strcat(strcpy(fullpath, rootdir), "expenseS.txt"), &expensesimsPtr);

      end = clock();
      printf("Finished simulations in %5d minutes %5d seconds \n ", (end-start)/CLOCKS_PER_SEC/60,
             ((end-start)/CLOCKS_PER_SEC)%60);
      fprintf(timeStamp, "Finished simulations in %5d minutes %5d seconds \n ",(end-start)/CLOCKS_PER_SEC/60,
             ((end-start)/CLOCKS_PER_SEC)%60);     
      fclose(timeStamp); //didn't open unless printOn==1.
   }

if (errorPrintOn==1) fclose(errorOutput);    //didn't open unless errorprintOn==1

   return 0;
}   /* End of main*/

/*--------------------------------------------------------------------------------*/
/*----------------------------------SUBROUTINES-----------------------------------*/

/* Flow Utility Function */

double getUtility(double cons) 
{
   double consAdj, utility;
   if (nu<0) {printf("getUtility(): passed nu==0, exiting.\n"); exit(1);} // divide by zero error coming up

   utility = -999; // to remove warning
   consAdj = c_0 + cons;
   if (nu==1.0) { utility = cscale*log(consAdj); }
   else if (nu == 0.0) { utility = cscale*cons; }
   else { utility = pow(consAdj,1-nu)*cscale/(1-nu); }

   return utility;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  net bequest function  */

double NetBequest(double amountBequestedP)
{
   double netBequest = 0; 
   if (amountBequestedP>exBeq)
      netBequest = exBeq+(1-tauBeq)*(amountBequestedP-exBeq); 
   else netBequest = amountBequestedP;
   if (netBequest<consFloor) { netBequest = consFloor; }
   return netBequest; 
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/* utility from a net bequest for a single person
   phi_j(b_net) = phi_j*( (b_net+K_j)^(1-nu) )/(1-nu) 
   not discounted */

double UBeqSingle(double netBequestP)
{
   double utils;    
   if (nu  ==  1)   /* log utility */
      utils  =  cscale*thetaB*log(netBequestP/(thetaB+1e-20)+c_1); 
   else
      utils  =  cscale*thetaB*pow(netBequestP/(thetaB+1e-20)+c_1, nu2)/nu2;         
   return utils; 
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/* Expected utility from leaving bequest matrix for a single, not discounted */

void GetUtilityBeq(double bequestUM[], double astate[])
{
   int aInd;
   for (aInd = 0; aInd<assetNum; aInd++)
      bequestUM[aInd] = UBeqSingle(NetBequest(astate[aInd]));
}

//*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/* Finding ideal igoods given capital stock and eTFP                              */

double getbaseIGoods(double capital, double eTFP, double gag, double agag, double ag3)
{
   double igoods;
   igoods = agag*pow(capital, gag)*pow(eTFP, ag3) - igshift;
   if (igoods < 0) { igoods = 0; }
   return igoods;
}
/*--------------------------------------------------------------------------------* /
/*--------------------------------------------------------------------------------*/
/* Output pre-TFP shift                                                           */

double getBaseRevenues(double capital, double igoods, double gamma, double ag2)
{
   return pow(capital, gamma)*pow(igoods+igshift,ag2);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/* After-tax income at bracket points, used to calculate income after tax*/

void IncomeAtBrk(double taxBrk[], double taxMar[], double incomeBrk[])
{
   int j; 

   incomeBrk[0] = (1-taxMar[0])*taxBrk[0]; /* The leftmost interval  */
   for (j = 1; j<(taxDim-1); j++)
   {
      incomeBrk[j] = incomeBrk[j-1]+(1-taxMar[j])*(taxBrk[j]-taxBrk[j-1]); 
   }
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/* Calculate income after tax 
   netIncome(j) = netIncome(j-1)+(1-taxMar(j))*(taxBrk(j)-taxBrk(j-1))  */

double AfterTaxIncome(double y)
{
   if (y<0)
   {
   /* printf("Error! negtive gross income!\n "); */
      return -1;  /* this case will be ruled out in maximization  */
   }
   else if (y<taxBrk[0])
      return ((1-taxMar[0])*(y)); 
   else if (y<taxBrk[1])
      return (incomeBrk[0]+(1-taxMar[1])*(y-taxBrk[0])); 
   else if (y<taxBrk[2])
      return (incomeBrk[1]+(1-taxMar[2])*(y-taxBrk[1])); 
   else if (y<taxBrk[3])
      return (incomeBrk[2]+(1-taxMar[3])*(y-taxBrk[2])); 
   else if (y<taxBrk[4])
      return (incomeBrk[3]+(1-taxMar[4])*(y-taxBrk[3])); 
   else if (y<taxBrk[5])
      return (incomeBrk[4]+(1-taxMar[5])*(y-taxBrk[4])); 
   else 
      return (incomeBrk[5]+(1-taxMar[6])*(y-taxBrk[5])); 
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/

double **getMarkovChain(double *chainInfoPtr, int numStates, double *rho, double *std, 
                        double *values, double *piInvarV)
{
   int iState, jState, kState, h, hMax;
   double **transmtx, **ProductMat, **subtot;

   *rho       = chainInfoPtr[1];
   *std       = chainInfoPtr[2];
   transmtx   = (double **)malloc(numStates*sizeof(double *));

   for(iState=0; iState<numStates; iState++)
   {
      values[iState]   = chainInfoPtr[3+iState];
     kState = iState;
     if (*rho == 0) { kState = 0; }
      transmtx[iState] = &chainInfoPtr[3+numStates*(kState+1)];
   }

// Now find Invariant distribution

   ProductMat = (double **)malloc(numStates*sizeof(double *));
   subtot     = (double **)malloc(numStates*sizeof(double *));
   for(iState=0; iState<numStates; iState++)
   {
      ProductMat[iState] = (double *)malloc(numStates*sizeof(double));
      subtot[iState]     = (double *)calloc(numStates,sizeof(double));
      for(jState=0; jState<numStates; jState++)
         ProductMat[iState][jState] = transmtx[iState][jState];
   }

   hMax = 2000;
   if (*rho==0) { hMax = 0; }

   for(h=0; h<hMax; h++)
   {
      for(iState=0; iState<numStates; iState++)
      {
         for(jState=0; jState<numStates; jState++)
         {
            subtot[iState][jState] = 0;

            for(kState=0; kState<numStates; kState++)          
               subtot[iState][jState] += ProductMat[iState][kState]*transmtx[kState][jState];

            ProductMat[iState][jState] = subtot[iState][jState];
         }
      }
   } 

   for(iState=0; iState<numStates; iState++)
      piInvarV[iState] = ProductMat[0][iState];

   return transmtx;
} 

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/

double *GetCDFvec(int nCols, double *probvec) 
{
   int iCol;
   double sum;
   double *CDFvec;

   CDFvec = (double *)malloc((nCols+1)*sizeof(double));
   
   sum = 0;   
   for (iCol=0; iCol<nCols; iCol++)
   {
      CDFvec[iCol] = sum;
      sum += probvec[iCol];
   }
   CDFvec[nCols] = sum + 1e-10;

return CDFvec;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/

double **GetCDFmtx(int nRows, int nCols, double **transmtx) 
{
   int iRow, iCol;
   double sum;
   double **transCDFmtx;

   transCDFmtx = (double **)malloc(nRows*sizeof(double *));
   for(iRow=0; iRow<nRows; iRow++)
      transCDFmtx[iRow] = (double *)malloc((nCols+1)*sizeof(double));
   
   for (iRow=0; iRow<nRows; iRow++)
   {
      sum = 0;   
      for (iCol=0; iCol<nCols; iCol++)
      {
         transCDFmtx[iRow][iCol] = sum;
         sum += transmtx[iRow][iCol];
      }
      transCDFmtx[iRow][nCols] = sum + 1e-10;
   }

return transCDFmtx;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  Locates nearest point _below_ x in a sorted array
    From Numerical Recipes in C, p. 117
*/

int Locate(double *Xarray, double x, int DIM)
{
   int j_L, j_U, j_M, ascend, dif, j_Star;

   ascend = 1;
   if (Xarray[DIM-1]<Xarray[0]) ascend = 0;

   j_L = 0;
   j_U = DIM-1;
   dif = j_U-j_L;

   if (ascend==1)
   {
      while (dif>1)
      {
         j_M = (int) (j_U+j_L)/2;
         if (x>Xarray[j_M])  { j_L = j_M; }
         else  { j_U = j_M; }
         dif = j_U-j_L;
      }
      j_Star = j_L;
   }

   else
   {
      while (dif>1)
      {
         j_M = (int) (j_U+j_L)/2;
         if (x<Xarray[j_M])  { j_L = j_M; }
         else { j_U = j_M; }
         dif = j_U-j_L;
      }
      j_Star = j_L;
   }

   return j_Star;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  Locates the point closest to x in a sorted array of size DIM
    Modified using the Locate function just above cris
*/
int LocateClosest(double *Xarray, double x, int DIM)
{
   int j_L, j_U, j_M, ascend, dif, j_Star;

   ascend = 1;
   if (Xarray[DIM-1]<Xarray[0]) { ascend = 0; }
   
   j_L = 0;
   j_U = DIM-1;
   dif = j_U-j_L;

   if (ascend==1)
    {
      while (dif>1)
      {
         j_M = (int) (j_U+j_L)/2;
         if (x>Xarray[j_M])  { j_L = j_M; }
         else  { j_U = j_M; }         
         dif = j_U-j_L;
      }
   // now find out the closest point to x, is it Xa(j_L) or Xa(j_L+1)?
   // remember that in the ascending case, by construction they are ordered as Xa(j_L) =< x =< Xa(j_L+1) 
      if ((x-Xarray[j_L]) <= (Xarray[j_L+1]-x))  { j_Star = j_L; }
      else  { j_Star = j_L+1; }   
   }
   else
    {
      while (dif>1);
      {
         j_M = (int) (j_U+j_L)/2;
         if (x<Xarray[j_M])  { j_L = j_M; }
         else  { j_U = j_M; }         
         dif = j_U-j_L;
      }      
   // now find out the closest point to x, is it Xa(j_L) or Xa(j_L+1)?
   // remember that in the descending case, by construction they are ordered as Xa(j_L) >= x >= Xa(j_L+1) 
      if ((x-Xarray[j_L+1]) > (Xarray[j_L]-x))  { j_Star = j_L; }
      else  { j_Star = j_L+1; } 
   }
   return j_Star;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/* Returns grid point and linear interpolation weight */
struct IndexInfo GetLocation(double *xP,  double x, int DIM)
{
   int j;
   double weight;
   struct IndexInfo IndexInfo1;

   j = Locate(xP,x,DIM)+1;
   IndexInfo1.Ind1=j-1;
   weight = (xP[j]-x)/(xP[j]-xP[j-1]);
   if (weight>1) { weight=1; }
   if (weight<0) { weight=0; }
   IndexInfo1.weight = weight;
   return IndexInfo1;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  Evaluate inner product for MV logit or preference shocks                      */
/*  JBJ:  01/30/12                                                                */

double XtimesBeta(double *coeffs, double age, int sexInd, double PInc, int healthStat)
{
   double totProduct;
   int iHS, i;

   totProduct = coeffs[0] + coeffs[1]*age + coeffs[2]*pow(age,2)/100 +
                coeffs[3]*pow(age,3)/10000 + (coeffs[4]+coeffs[6]*age)*PInc +
                coeffs[5]*pow(PInc,2);
   if (sexInd==0)
      totProduct += (coeffs[7]+coeffs[8]*age);  // Male


   for (iHS=1; iHS<HSDIM; iHS++) // Health Dummies
   {
      if (healthStat==iHS)
      {
         i = 9+(iHS-1)*2;
         totProduct += (coeffs[i]+coeffs[i+1]*age);
      }
   }

   return(totProduct);
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  Use MV logit model to get mortality and health status transition probs        */
/*  JBJ:  01/31/12                                                                */

double GetTransProbs(double **hsCoeffs, double age, int sexInd, 
                     double PInc, int healthStat, double HSProbs[])
{
   double totProduct, MVLsum, survprob;
   int iHS;

   MVLsum = 1; // Death is the benchmark state

   for (iHS=0; iHS<HSDIM; iHS++) // iHS indexes future health, healthStat indexes current health
   {
      totProduct   = XtimesBeta(hsCoeffs[iHS], age, sexInd, PInc, healthStat);
      HSProbs[iHS] = exp(totProduct);
      MVLsum      += exp(totProduct);
   }

   survprob = 1 - 1/MVLsum;

   for (iHS=0; iHS<HSDIM; iHS++)
   {
      HSProbs[iHS] = HSProbs[iHS]/MVLsum;
      HSProbs[iHS] = HSProbs[iHS]/survprob; // Probability conditional on surviving
   }

   return(survprob);
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  LOGIT  */

double Logit(double x)
{
   return exp(x)/(1+exp(x));
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  PRODFNPARMS */

void prodFnParms(double alpha0, double gamma0, double *alpha, double *gamma, 
                 double *ag2, double *ag3, double *gag, double *agag)
{  
// All these parameters are globals
   *alpha = alpha0;
   *gamma = gamma0;
   *ag2   = 1 - *alpha - *gamma;
   *ag3   = 1 / (*alpha + *gamma);
   *gag   = (*gamma)*(*ag3);
   *agag  = pow(*ag2, *ag3);   
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  LOGITSQRT  */

double LogitSQRT(double x)
{
   return sqrt(exp(x) / (1 + exp(x)));
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  GLOBDEREF:  Dereference a bunch of pointers to C++ globals  */

int globderef(double *agevecPtr, double *sizevecPtr, double *prefvecPtr, double *finvecPtr, 
              double *initWage)
{
   char fullpath[ADDRESS_LEN];

   bornage       = (int) floor(rounder+agevecPtr[0]);
   retage        = (int) floor(rounder+agevecPtr[1]);
   lifespan      = (int) floor(rounder+agevecPtr[2]);
   firstyear     = (int) floor(rounder+agevecPtr[3]);
   lastyear      = (int) floor(rounder+agevecPtr[4]);
   timespan      = (int) floor(rounder+agevecPtr[5]);
   numSims       = (int) floor(rounder+agevecPtr[6]);

   assetMin      = sizevecPtr[0];
   assetMax      = sizevecPtr[1];
   equityMin     = sizevecPtr[2];
   equityMax     = sizevecPtr[3];
   totassetMin   = sizevecPtr[4];
   totassetMax   = sizevecPtr[5];
   debtMin       = sizevecPtr[6];
   debtMax       = sizevecPtr[7];
   capitalMin    = sizevecPtr[8];
   capitalMax    = sizevecPtr[9];
   cashMin       = sizevecPtr[10];
   cashMax       = sizevecPtr[11];

// Flow utility parameters:
//   beta = discount factor
//   u(c) = ((c_0+c)^(1-nu))/(1-nu)
//   V_{T+1} = ((c_1+a/thetaB)^(1-nu))*thetaB

   beta          = prefvecPtr[0];
   nu            = prefvecPtr[1];
   nu2           = prefvecPtr[2];
   c_0           = prefvecPtr[3];
   c_1           = prefvecPtr[4];
   thetaB        = prefvecPtr[5];
   chi           = prefvecPtr[6];  // Here it is a consumption increment
   consFloor     = prefvecPtr[7];
   cscale        = pow(c_0+initWage[0],-nu2);
   chi           = cscale*(1/nu2)*(pow(c_0+initWage[0],nu2)- pow(c_0+initWage[0]-chi,nu2));  // Here it is a utility increment

// Production function and finance constraints
//  Y = z(M^alp)(K^gam)(N^[1-alp-gam])
//  G*a_t = (1-delta+gk)k_{owned,t-1) + y_{t-1} - (x_{t-1} - a_{t-1} - b_t-`

   alpha1        = finvecPtr[0];
   alpha2        = finvecPtr[1];
   gamma1        = finvecPtr[2];
   gamma2        = finvecPtr[3];
   igshift       = finvecPtr[4];
   lambda        = finvecPtr[5];
   phi           = finvecPtr[6];
   zeta          = finvecPtr[7];
   delta         = finvecPtr[8];
   fixedcost     = finvecPtr[9];
   psi_inverse   = finvecPtr[10];  // fraction of capital that can be borrowed, inverted.  Negative => no limit
   r_riskfree    = finvecPtr[11];
   bigR          = finvecPtr[12];
   bigG          = finvecPtr[13];
   eGK           = finvecPtr[14]; // Expected value
   noRenegotiation = (int) floor(rounder+finvecPtr[15]);
   ftNum         = (int)floor(rounder + finvecPtr[16]);
   eRDG          = r_riskfree+delta-eGK;
   SLfrac        = phi*(1-delta+eGK)/bigG;  // irreversibility costs for t-1 capital

   if (printOn>0) 
   {
      parmValues = fopen(strcat(strcpy(fullpath, outputdir), "c_parmValues.txt"), "w");
      fprintf(parmValues, "alpha1: %f\nalpha2: %f\ngamma1: %f\ngamma2: %f\nIGoodShift: %f\nlambda: %f\nphi: %f\nzeta: %f\ndelta: %f\nfixedcost: %f\npsi_inverse: %f\nr_riskfree: %f\nbigG: %f\neGK: %f\n", 
              alpha1, alpha2, gamma1, gamma2, igshift, lambda, phi, zeta, delta, fixedcost, psi_inverse, r_riskfree, bigG, eGK);
      fprintf(parmValues, "beta: %f\nnu: %f\nc_0: %f\nc_1: %f\nthetaB: %f\nchi: %f\nconsFloor: %f\ncScaler: %f\n", 
              beta, nu, c_0, c_1, thetaB, chi, consFloor, cscale);
      fprintf(parmValues, "bornage: %d\nretage: %d\nfirstyear: %d\nlastyear: %d\nnumSims: %d\n", 
              bornage, retage, firstyear, lastyear, numSims);
      fprintf(parmValues, "assetNum: %d\nequityNum: %d\ntotassetNum: %d\ndebtNum: %d\nnumThreads: %d\n", 
              assetNum, equityNum, totassetNum, debtNum, size);
      fclose(parmValues);
   }

   return 1;
} 

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  GETASSIGNMENTVEC:  Allocate points on a state vector across nodes             */
/*                     Uses Sangeeta Pratap's allocation trick                    */

int *getAssignmentVec(int numPoints, int numNodes)
{
   int numBlocks, iBlock, iNode, iPoint, thisPoint;
   double nb2;
   int *assignmentVec;

   assignmentVec = (int *)calloc(numPoints,sizeof(int));

// for (iPoint=0; iPoint<numPoints; iPoint++)
//    assignmentVec[iPoint] = iPoint;

   nb2           = ((double) numPoints)/ ((double) numNodes);
   numBlocks     = (int) ceil(nb2);  
   iPoint        = 0;

// Now scramble, by pulling numbers from different "blocks" 
   for (iNode=0; iNode<numNodes; iNode++)
   {
      for (iBlock=0; iBlock<numBlocks; iBlock++)
      {
         thisPoint = iBlock*numNodes+iNode+1;
         if (thisPoint>numPoints) { continue; }   // Recall that indexing starts at zero
         assignmentVec[iPoint] = thisPoint-1;
         iPoint += 1;
      }
   }  
   return assignmentVec;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  GETCLOSESTLK:  The lagged capital state vector may be coarser than the vector */
/*                 for capital choices.  Find the index of the lagged capital     */
/*                 value closest to each element of the capital choice vector.    */

void getclosestLK(int numPTrue, int numPTarget, double *truevec, double *targetvec, 
                  int *CLKindvec, double *CLKwgtvec)
{
   int iTV;
   struct IndexInfo thisLocation;

   for (iTV = 0; iTV < numPTrue; iTV++)
   {
      thisLocation   = GetLocation(targetvec, truevec[iTV], numPTarget);
      CLKindvec[iTV] = thisLocation.Ind1;
      CLKwgtvec[iTV] = thisLocation.weight;
   }
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/* Solve value function for t = 1, ..., T   
   Worker's case 
*/
void GetRulesWorker(int iAssetsmin, int iAssetsmax, double *assetvec, double *wageProfile, 
                    double **valfuncWork, double **bestCWork, double **bestNPIWork)
{
   int tInd, aInd, aNPInd, maxNPI, NPIlim, numSteps, aNPImin, aNPImax;

   double cashonhand, diff, todaysUtility, cons, maxCons, continuationValue, value, 
          maxValue, oldMaxV, oldoldMaxV, stepPct, numStepsF; 

   for (tInd = (lifespan-1); tInd>= 0; tInd--)
   {  
      printf("tInd %5d\n", tInd);
      for (aInd = iAssetsmin; aInd<iAssetsmax; aInd++) 
      {               
      // Compute cash-on-hand (GROSS OF TAXES), before any transfers
         cashonhand = bigR*assetvec[aInd] + wageProfile[tInd]; // assetvec[aInd] + wageProfile[tInd];
         NPIlim = 0;

         for (aNPInd=0; aNPInd<assetNum; aNPInd++)
         {
            diff = cashonhand - bigG*assetvec[aNPInd] - consFloor; // Use to impose that consumption is greater than consFloor
            if (diff<0 ) {break;} 
         }  

         NPIlim = aNPInd; // maximum feasible saving + 1
         if (NPIlim < 1) NPIlim = 1; 

      // Initialize maximum to a value that can be exceeded for sure
         todaysUtility   = getUtility(consFloor/2 - c_0/2); 
         continuationValue = valfuncWork[tInd+1][0];
         value           = todaysUtility + beta*continuationValue;
         maxValue        = value - 1e6; 
         oldMaxV         = maxValue;
         oldoldMaxV      = oldMaxV;
         maxCons         = -1;
         maxNPI          = -1;
         aNPImin         = 0;
         aNPImax         = NPIlim;
         
         if (tInd<(lifespan-1))
         {
            stepPct      = minStepPct + pow((1-gridShrinkRate),(lifespan-1-tInd));
            numStepsF    = stepPct*assetNum;
            numSteps     = (int) numStepsF;
            aNPImin      = (int) bestNPIWork[tInd+1][aInd] - numSteps;
            aNPImax      = aNPImin + 2*numSteps + 1;
            if (aNPImin<0) { aNPImin = 0; }
            if (aNPImax>NPIlim) { aNPImax = NPIlim; }
         }
           
         for (aNPInd=aNPImin; aNPInd<aNPImax; aNPInd++)
         {
         // loop over decision variable, all possible savings levels tomorrow 
            cons              = cashonhand - bigG*assetvec[aNPInd];
            if (cons < consFloor) { cons = consFloor; }
            todaysUtility     = getUtility(cons); 
            continuationValue = valfuncWork[tInd+1][aNPInd];
            value             = todaysUtility + beta*continuationValue;

            if (value>maxValue)
            {
               oldoldMaxV     = oldMaxV;
               oldMaxV        = maxValue; // record previous maximums      
               maxValue       = value;
               maxCons        = cons;
               maxNPI         = aNPInd;
            }
         // If value function is decreasing in savings for two consecutive times we quit.
         // Makes sense if objective is concave.
            if ((value < oldMaxV) && (value < oldoldMaxV) && (aNPInd>3)) { break; }
         }

         valfuncWork[tInd][aInd] = maxValue;
         bestCWork[tInd][aInd]   = maxCons;
         bestNPIWork[tInd][aInd] = maxNPI;
      } // end loop over current assets 
      
  //  Wait until every processor has hit this point.
      #pragma omp barrier

   } //  (for (tInd = (lifespan-1); tInd>= 0; tInd--)), loop over age

// printf("valueFunM_0s = %5.2lf\n", valueFunM[0][0][0][0][0][0][0]);

}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  GETLIQINDEX:  Find point on asset grid closest to post-liquidation net worth  */

void getliqIndex(double *assetvec, double *totassetvec, double *lagcapvec, 
                 double *debtvec, double ***postliqAssets, double ***postliqNetWorth, 
                 double ***postliqNWIndex)
{
   int iTotAsset, iLagK, iDebt, closestInd;
   double thisAsset, sellingLoss, thisNetWorth;

   for (iLagK=0; iLagK<lagcapNum; iLagK++)
   {
      sellingLoss = SLfrac*lagcapvec[iLagK]; // If there are capital irreversibilities 
      for (iTotAsset=0; iTotAsset<totassetNum; iTotAsset++)
      {      
         thisAsset   = (1-lambda)*(totassetvec[iTotAsset]-sellingLoss);
         if (thisAsset<0) { thisAsset = 0; } // Firm may experience massive operating loss

         for (iDebt=0; iDebt<debtNum; iDebt++)
         {       
            thisNetWorth = thisAsset - debtvec[iDebt]; // Debt is defined to be non-negative
            if (thisNetWorth<0) { thisNetWorth = 0; }
            closestInd   = LocateClosest(assetvec, thisNetWorth, assetNum);
            postliqAssets[iLagK][iTotAsset][iDebt]   = thisAsset;
            postliqNWIndex[iLagK][iTotAsset][iDebt]  = closestInd;
            postliqNetWorth[iLagK][iTotAsset][iDebt] = assetvec[closestInd];
         }
      }
   }
   return;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  GETNPTOTASSETS:  Go over farmer's decision grid, and figure out future total  */
/*                   assets associated with each decision.                        */
/*                   Since dividends are the residual to the budget constraint,   */
/*                   These calculations need only be done once.                   */
/*                   Note that capital adjustment costs are subtracted from the   */
/*                   that were issued PRIOR to these production sequences.        */

void getNPtotAssets(int iTotKmin, int iTotKmax, double *capitalvec, double *NKratiovec, 
                    double *cashvec, double *feValues, double *zValues, double *totassetvec, 
                    double ******NPtotassetWeight, int ******NPtotassetIndex, 
                    int *****goodGridPoint)
{
   int iTotK, iNKrat, iCash, iFType, iFE, iZNP;
   struct IndexInfo thisPoint;
   double alpha, gamma, ag2, ag3, gag, agag, totCapital, baseRevenues, baseIGoods, 
          igoods, expenses, thisCash, totAssetsNP;
      
   for (iFType = 0; iFType<ftNum; iFType++)
   {
      if (iFType==0) { prodFnParms(alpha1, gamma1, &alpha, &gamma, &ag2, &ag3, &gag, &agag); } // only locals are being updated
      else { prodFnParms(alpha2, gamma2, &alpha, &gamma, &ag2, &ag3, &gag, &agag); }

      for (iTotK=iTotKmin; iTotK<iTotKmax; iTotK++)
      {
         totCapital = capitalvec[iTotK];
         for (iFE = 0; iFE<feNum; iFE++)
         {
            baseIGoods = getbaseIGoods(totCapital, feValues[iFE], gag, agag, ag3);
            for (iNKrat = 0; iNKrat<NKratioNum; iNKrat++)
            {       
               igoods       = baseIGoods*NKratiovec[iNKrat];
               baseRevenues = getBaseRevenues(totCapital, igoods, gamma, ag2);
               expenses     = igoods + fixedcost;

               for (iCash=cashNum-1; iCash>=0; iCash--)
               {       
                  thisCash = cashvec[iCash];
                  if (thisCash < expenses/zeta) { break; } // Check cash-in-advance constraint;
                  goodGridPoint[iTotK][iFType][iFE][iNKrat][iCash] = 1;

                  for (iZNP=0; iZNP<zNum; iZNP++)
                  {       
                     totAssetsNP = ( (1-delta+eGK)*totCapital + feValues[iFE]*zValues[iZNP]*baseRevenues - expenses + thisCash )/bigG;
                     thisPoint   = GetLocation(totassetvec, totAssetsNP, totassetNum);
                     NPtotassetIndex[iTotK][iFType][iFE][iNKrat][iCash][iZNP]  = thisPoint.Ind1;
                     NPtotassetWeight[iTotK][iFType][iFE][iNKrat][iCash][iZNP] = thisPoint.weight;
                  } 
               }
            } 
         }
      }
   }
   return;
}
     
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  GETFINALLIQ:  Calculate debt repaid at retirement.  Fill in other vectors.    */
/*                Find retirement utility.                                        */

void getFinalLiq(int iTotAssetmin, int iTotAssetmax, double *totassetvec, 
                 double *lagcapvec, double *debtvec, double *assetvec,  
                 double ***retNWIndex,double *******liqDecisionMat, 
                 double *******valfuncMat, double *******fracRepaidMat) 
{
   int iTotAsset, iLagK, iDebt, iZ, iFType, iFE;
   double sellingLoss, thisAsset, thisDebt, finalAsset, newDebt, thisUtil, thisFrac;

   for (iLagK=0; iLagK<lagcapNum; iLagK++)
   {
      sellingLoss = SLfrac*lagcapvec[iLagK]; // If there are capital irreversibilities
      for (iTotAsset=iTotAssetmin; iTotAsset<iTotAssetmax; iTotAsset++)
      {
         thisAsset   = totassetvec[iTotAsset] - sellingLoss;  // No liquidation costs for retirement
         if (thisAsset<0) { thisAsset = 0; }  // Firm may experience massive operating loss        
       
         for (iDebt=0; iDebt<debtNum; iDebt++)
         { 
            thisDebt   = debtvec[iDebt];
            newDebt    = thisDebt;
            if (newDebt>thisAsset) { newDebt = thisAsset; }
            finalAsset = thisAsset - newDebt;
            retNWIndex[iLagK][iTotAsset][iDebt] = LocateClosest(assetvec, finalAsset, assetNum);
            thisUtil   = UBeqSingle(NetBequest(finalAsset));
            thisFrac   = 1;
            if (thisDebt>0) { thisFrac = newDebt/thisDebt; }
            for (iFType=0; iFType<ftNum; iFType++)
            {
               for (iFE=0; iFE<feNum; iFE++)
               {
                  for (iZ = 0; iZ<zNum2; iZ++)
                  {
                     valfuncMat[lifespan][iFType][iFE][iZ][iLagK][iTotAsset][iDebt]     = thisUtil;
                     liqDecisionMat[lifespan][iFType][iFE][iZ][iLagK][iTotAsset][iDebt] = 1;
                     fracRepaidMat[lifespan][iFType][iFE][iZ][iLagK][iTotAsset][iDebt]  = thisFrac;
                  }
               }
            }
         }
      }
   }  
   return;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  GETLIQDECISION:  Find debt load that makes family indifferent between liqui-  */
/*                   dation and continuing to farm.  See if this implies liqui-   */
/*                   dation and calculate repayment/obligation ratio.             */
/*                   Find continuation utility of optimal choice                  */

void getliqDecision(double *totassetvec, double *debtvec, double *equityvec,  
                    double ***postliqAssets, double ***postliqNWIndex, double **valfuncWork,  
                    double ******valfuncFarm, double *******liqDecisionMat, 
                    double *******valfuncMat, double *******fracRepaidMat,
                    int iTotAssetMin, int iTotAssetMax, int ageInd) 
{
   int iTotAsset, iLagK, iDebt, iFType, iFE, iZ, lowInd, iLiq, liqDec;
   struct IndexInfo thisPoint;
   double thisAsset, thisDebt, liqValue, lowWeight, thisEquity, newDebt, contValue, thisVal, thisFrac;

   for (iLagK=0; iLagK<lagcapNum; iLagK++)
   {
      for (iTotAsset=iTotAssetMin; iTotAsset<iTotAssetMax; iTotAsset++)
      {
         thisAsset = totassetvec[iTotAsset];
         
         for (iDebt=0; iDebt<debtNum; iDebt++)
         {       
            thisDebt = debtvec[iDebt];
         // First find lifetime utility from liquidation
            iLiq     = (int) postliqNWIndex[iLagK][iTotAsset][iDebt]; 
            liqValue = valfuncWork[ageInd][iLiq]; // Don't interpolate here, this asset grid is fine

            if (noRenegotiation==1) // Automatically liquidate upon default
            {
               thisEquity = thisAsset - thisDebt;

               if (thisEquity<0) // Must liquidate
               {
                  liqDec     = 1;                     
               // Find lifetime utility from liquidation
                  thisVal    = liqValue;
                  thisFrac   = 1;
                  newDebt    = postliqAssets[iLagK][iTotAsset][iDebt];  // Note:  Postliquidation assets <= thisAsset < thisDebt
                  if (thisDebt>0) { thisFrac = newDebt / thisDebt; }
               }
               else
               {
                  liqDec     = 0;                     
                  thisFrac   = 1;
                  thisPoint  = GetLocation(equityvec, thisEquity, equityNum);
                  lowInd     = thisPoint.Ind1;
                  lowWeight  = thisPoint.weight;
               }

               for (iFType=0; iFType<ftNum; iFType++)
               {
                  for (iFE=0; iFE<feNum; iFE++)
                  {
                     for (iZ=0; iZ<zNum2; iZ++)
                     {
                        if (liqDec<1) // Can still CHOOSE to liquidate
                        {
                           thisVal = valfuncFarm[ageInd][iFType][iFE][iZ][iLagK][lowInd]*lowWeight 
                                     + valfuncFarm[ageInd][iFType][iFE][iZ][iLagK][lowInd+1]*(1-lowWeight);

                     if (thisVal < liqValue)
                           { 
                              liqDec     = 1;   
                              thisVal    = liqValue;
                              thisFrac   = 1;
                              newDebt    = postliqAssets[iLagK][iTotAsset][iDebt];  // Note:  Postliquidation assets <= thisAsset < thisDebt
                              if (thisDebt>0) { thisFrac = newDebt / thisDebt; }
                           }
                        }
                        liqDecisionMat[ageInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt] = (float)liqDec;
                        fracRepaidMat[ageInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt]  = thisFrac;
                        valfuncMat[ageInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt]     = thisVal;
                     }
                  }  // end loop through fixed effect TFP shocks
               }  // end loop through production types
            } // End no-renegotiation branch 

            else  // Consider renegotiation
            {
               for (iFType = 0; iFType<ftNum; iFType++)
               {
                  for (iFE=0; iFE<feNum; iFE++)
                  {
                     for (iZ=0; iZ<zNum2; iZ++)
                     {
                        valfuncMat[ageInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt] = liqValue;

                     // Now find operating equity that yields same lifetime utility.
                        thisPoint     = GetLocation(valfuncFarm[ageInd][iFType][iFE][iZ][iLagK], liqValue, equityNum); // Equity grid is coarser, gotta interpolate
                        lowInd        = thisPoint.Ind1;
                        lowWeight     = thisPoint.weight;
                        thisEquity    = equityvec[lowInd]*lowWeight + equityvec[lowInd+1]*(1-lowWeight);
                        newDebt       = thisAsset - thisEquity;  // Farmer will sacrifice this much to avoid liquidation
                        liqDec        = 0;

                        if (newDebt>thisDebt) // No renegotiation
                        {
                           newDebt    = thisDebt;
                           thisEquity = thisAsset - thisDebt;
                           thisPoint  = GetLocation(equityvec, thisEquity, equityNum);
                           lowInd     = thisPoint.Ind1;
                           lowWeight  = thisPoint.weight;
                           contValue  = valfuncFarm[ageInd][iFType][iFE][iZ][iLagK][lowInd]*lowWeight 
                                        + valfuncFarm[ageInd][iFType][iFE][iZ][iLagK][lowInd+1]*(1-lowWeight);
                           valfuncMat[ageInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt] = contValue;
                        }
                        else if (newDebt<postliqAssets[iLagK][iTotAsset][iDebt]) // Liquidation
                        {
                           liqDec = 1;
                           newDebt = postliqAssets[iLagK][iTotAsset][iDebt];
                           if (newDebt>thisDebt) { newDebt = thisDebt; }
                        }

                        liqDecisionMat[ageInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt] = (float)liqDec;
                        fracRepaidMat[ageInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt] = 1;
                        if (thisDebt>0) { fracRepaidMat[ageInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt] = newDebt / thisDebt; }

                     }  // end loop through transitory TFP shocks
                  }  // end loop through fixed effect TFP shocks
               }  // end loop through production types
            }  // End renegotiation branch
         }
      }
   }
   return;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
void getOperatingDec(int jEquityMin, int jEquityMax, int *eqAssignvec, int ageInd, 
                     double *equityvec, double *capitalvec, double *lagcapvec, double *debtvec, 
                     double *NKratiovec, double *cashvec, double **zTransmtx, int *CLKindvec, 
                     double *CLKwgtvec,
                     int *****goodGridPoint, double ******NPtotassetWeight, int ******NPtotassetIndex, 
                     double *******valfuncMat, double *******fracRepaidVec, 
                     double ******valfuncFarm, double ******bestIntRateFarm, int ******bestKIndexFarm, 
                     int ******bestNKratIndexFarm, int ******bestCashIndexFarm, 
                     double ******bestDividendFarm, int ******bestDebtIndexFarm, 
                     double *******bestNPTAWeightFarm, int*******bestNPTAIndexFarm, 
                     double ******bestKFarm, double ******bestNKratFarm, double ******bestDebtFarm, 
                     double ******bestCashFarm) 
{
   int iFType, iFE, iZ, iLagK, iEquity, jEquity, iZNP, iTotK, iNKrat, iCash, iDebt, bestTotKI, bestNKratI, 
       bestCashI, bestDebtI, numSteps, iTotKmin, iTotKmax, iNKratmin, iNKratmax, iCashmin, iCashmax, iDebtmin, 
       iDebtmax;
   double todaysUtility, continuationValue, value, maxValue, bestIR, bestDiv, salesloss, LOA, minDebt, thisDebt, 
          minRepay, bestTotK, bestNKrat, bestDebt, bestCash, thisLoan, thisDiv, eRepay, stepPct, numStepsF;
   double *bestNPTAW; // vector, one element for each potential z_t+1
   int *bestNPTAI;

   stepPct  = minStepPct + pow((1-gridShrinkRate),(lifespan-1-ageInd));
   minRepay = 0;
   if (ageInd>(lifespan-2)) { minRepay = eRepayMin; } // Rule out pathological borrowing behavior at the end of life

   for (iFType=0; iFType<ftNum; iFType++)
   {
      for (iFE=0; iFE<feNum; iFE++)
      {
         for (iZ = 0; iZ < zNum2; iZ++)
         {     
            for (iLagK=0; iLagK<lagcapNum; iLagK++)
            {    
               for (jEquity = jEquityMin; jEquity < jEquityMax; jEquity++)
               {
               // Initialize maximum to a value that can be exceeded for sure
                  iEquity    = eqAssignvec[jEquity];
                  todaysUtility = getUtility(consFloor/2 - c_0/2);
                  continuationValue = valfuncMat[ageInd+1][iFType][0][0][lagcapNum-1][0][debtNum-1];
                  value      = todaysUtility + beta*continuationValue;
                  maxValue   = value - 1e6;
                  bestDebtI  = -1;
                  bestDebt   =  0;
                  bestIR     = bigR;
                  bestTotKI  = -1;
                  bestTotK   =  0;
                  bestNKratI = -1;
                  bestNKrat  =  0;
                  bestCashI  = -1;
                  bestCash   =  0;
                  bestDiv    =  0;
                  bestNPTAW  = bestNPTAWeightFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity];
                  bestNPTAI  = bestNPTAIndexFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity];
                  for (iZNP = 0; iZNP < zNum; iZNP++)
                  {
                     bestNPTAW[iZNP] = -1;
                     bestNPTAI[iZNP] = -1;
                  }

                  iTotKmin  = 0;
                  iTotKmax  = capitalNum;
                  iNKratmin = 0;
                  iNKratmax = NKratioNum;
                  iCashmin  = 0;
                  iCashmax  = cashNum;
                  iDebtmin  = 0;
                  iDebtmax  = debtNum;

                  if ((ageInd<(lifespan-1)) && (bestKIndexFarm[ageInd+1][iFType][iFE][iZ][iLagK][iEquity]>-1))
                  {
                     numStepsF = stepPct*capitalNum;
                     numSteps = (int)numStepsF;
                     if (numSteps<2) { numSteps=2; }
                     iTotKmin = bestKIndexFarm[ageInd+1][iFType][iFE][iZ][iLagK][iEquity] - numSteps;
                     iTotKmax = iTotKmin + 2*numSteps + 1;
                     if (iTotKmin<0) { iTotKmin = 0; }
                     if (iTotKmax>capitalNum) { iTotKmax = capitalNum; }

                     numStepsF = stepPct*NKratioNum;
                     numSteps = (int)numStepsF;
                     if (numSteps<2) { numSteps = 2; }
                     iNKratmin = bestNKratIndexFarm[ageInd+1][iFType][iFE][iZ][iLagK][iEquity] - numSteps;
                     iNKratmax = iNKratmin + 2*numSteps + 1;
                     if (iNKratmin<0) { iNKratmin = 0; }
                     if (iNKratmax>NKratioNum) { iNKratmax = NKratioNum; }
   
                     numStepsF = stepPct*cashNum;
                     numSteps = (int)numStepsF;
                     if (numSteps<2) { numSteps = 2; }
                     iCashmin = bestCashIndexFarm[ageInd+1][iFType][iFE][iZ][iLagK][iEquity] - numSteps;
                     iCashmax = iCashmin + 2*numSteps + 1;
                     if (iCashmin<0) { iCashmin = 0; }
                     if (iCashmax>cashNum) { iCashmax = cashNum; }

                     numStepsF = stepPct*debtNum;
                     numSteps = (int)numStepsF;
                     if (numSteps<2) { numSteps = 2; }
                     iDebtmin = bestDebtIndexFarm[ageInd+1][iFType][iFE][iZ][iLagK][iEquity] - numSteps;
                     iDebtmax = iDebtmin + 2*numSteps + 1;
                     if (iDebtmin<0) { iDebtmin = 0; }
                     if (iDebtmax>debtNum) { iDebtmax = debtNum; }
                  }

                  for (iTotK = iTotKmin; iTotK < iTotKmax; iTotK++)
                  {
                     salesloss = (1-delta+eGK)*lagcapvec[iLagK]/bigG - capitalvec[iTotK];
                     if (salesloss>0) { salesloss = phi*salesloss; }
                     else { salesloss = 0; }
                              
                     for (iNKrat = iNKratmin; iNKrat < iNKratmax; iNKrat++)
                     {
                        for (iCash = iCashmax - 1; iCash >= iCashmin; iCash--)
                        {
                        // Check cash-in-advance constraint
                           if (goodGridPoint[iTotK][iFType][iFE][iNKrat][iCash] == 0) { break; }

                        // Impose non-negative dividend condition on debt
                           LOA = equityvec[iEquity] - capitalvec[iTotK] - cashvec[iCash] - salesloss;
                           minDebt = -(1 + r_riskfree)*(LOA+c_0*eqInject)/ bigG;

                           for (iDebt = iDebtmax - 1; iDebt >= iDebtmin; iDebt--)
                           {
                              thisDebt = debtvec[iDebt];
                              if (thisDebt<minDebt) { break; }
                              if ((thisDebt*psi_inverse) > capitalvec[iTotK]) { continue; } // Exogenous collateral constraint
                           // Now adjust loan for repayment risk
                              eRepay = getExpectation(fracRepaidVec[ageInd+1][iFType][iFE], NPtotassetIndex[iTotK][iFType][iFE][iNKrat][iCash],
                                                      NPtotassetWeight[iTotK][iFType][iFE][iNKrat][iCash], CLKindvec[iTotK], CLKwgtvec[iTotK], 
                                                      iDebt, zTransmtx[iZ]);
                              if ((eRepay <= minRepay) && (thisDebt>0)) { continue; }
                              if (eRepay > 1) { eRepay = 1; }
                              thisLoan = eRepay*thisDebt * bigG / (1 + r_riskfree);
                              thisDiv = LOA + thisLoan;
                              if (thisDiv<(consFloor-c_0*eqInject)) { continue; }

                              todaysUtility = getUtility(thisDiv) + chi;
                              continuationValue = getExpectation(valfuncMat[ageInd+1][iFType][iFE], NPtotassetIndex[iTotK][iFType][iFE][iNKrat][iCash],
                                                                 NPtotassetWeight[iTotK][iFType][iFE][iNKrat][iCash], CLKindvec[iTotK], CLKwgtvec[iTotK],
                                                                 iDebt, zTransmtx[iZ]);
                              value = todaysUtility + beta*continuationValue;

                              if (value>maxValue)
                              {
                                 maxValue   = value;
                                 bestDebtI  = iDebt;
                                 if (debtvec[iDebt] == 0) { eRepay = 1; }
                                 bestIR     = (1 + r_riskfree) / eRepay;
                                 bestTotKI  = iTotK;
                                 bestNKratI = iNKrat;
                                 bestCashI  = iCash;
                                 bestDiv    = thisDiv;
                                 bestTotK   = capitalvec[iTotK];
                                 bestNKrat  = NKratiovec[iNKrat];
                                 bestDebt   = debtvec[iDebt];
                                 bestCash   = cashvec[iCash];
                              }
                           }  // End loop through debt choices
                        }  // End loop through cash choices
                     }  // End loop through own capital choices       
                  }  // End loop through total capital choices

                  valfuncFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]        = maxValue;
                  bestCashIndexFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]  = bestCashI;
                  bestDebtIndexFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]  = bestDebtI;
                  bestIntRateFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]    = bestIR;
                  bestKIndexFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]     = bestTotKI;
                  bestNKratIndexFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity] = bestNKratI;
                  bestDividendFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]   = bestDiv;
                  bestKFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]          = bestTotK;
                  bestNKratFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]      = bestNKrat;
                  bestDebtFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]       = bestDebt;
                  bestCashFarm[ageInd][iFType][iFE][iZ][iLagK][iEquity]       = bestCash;

                  if (bestTotKI > -1)
                  {
                     for (iZNP = 0; iZNP < zNum; iZNP++)
                     {
                        bestNPTAW[iZNP] = NPtotassetWeight[bestTotKI][iFType][iFE][bestNKratI][bestCashI][iZNP];
                        bestNPTAI[iZNP] = NPtotassetIndex[bestTotKI][iFType][iFE][bestNKratI][bestCashI][iZNP];
                     }
                  }
               }  // End loop through equity
            }  // End loop  through lagged capital
         }  // End loop through transitory TFP
      }  // End loop through FE productivity
   }  // End loop through production function type

   return;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
double getExpectation(double ****RandVar, int *NPTAIndex, double *NPTAWeight,
                      int iLagK, double iLagKwgt, int iDebt, double *zProbs)
{
   int iZNP, iZNP2, iNPTA;
   double expectsum, avg, avg2;

   expectsum = 0;

   if (lagcapNum==1)
   {
      for (iZNP = 0; iZNP<zNum; iZNP++)
      {
         iNPTA = NPTAIndex[iZNP];
      // Need to distinguish between TFP shock as it affects output (iZNP) and as a state variable (iZNP2)
         iZNP2 = iZNP;
         if (zNum2 == 1) { iZNP2 = 0; }
         avg = NPTAWeight[iZNP]*RandVar[iZNP2][0][iNPTA][iDebt]
               + (1-NPTAWeight[iZNP])*RandVar[iZNP2][0][iNPTA+1][iDebt];
         expectsum = expectsum + zProbs[iZNP]*avg;
      }
   }
   else
   {
      for (iZNP = 0; iZNP<zNum; iZNP++)
      {
         iNPTA = NPTAIndex[iZNP];
      // Need to distinguish between TFP shock as it affects output (iZNP) and as a state variable (iZNP2)
         iZNP2 = iZNP;
         if (zNum2 == 1) { iZNP2 = 0; }
         avg  = NPTAWeight[iZNP]*RandVar[iZNP2][iLagK][iNPTA][iDebt]
                + (1-NPTAWeight[iZNP])*RandVar[iZNP2][iLagK][iNPTA+1][iDebt];
         avg2 = NPTAWeight[iZNP]*RandVar[iZNP2][iLagK+1][iNPTA][iDebt]
                + (1-NPTAWeight[iZNP])*RandVar[iZNP2][iLagK+1][iNPTA+1][iDebt];
         avg  = avg*(iLagKwgt) + avg2*(1-iLagKwgt);
         expectsum = expectsum + zProbs[iZNP]*avg;
      }
   }

   return expectsum;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
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
                int iSimmin, int iSimmax)
{
   int personInd, yearInd, ageInd, ftInd, feInd, zInd, zInd2, lkInd, lkInd2, taInd, dInd, eqInd; 
   int age0, age, year0, liqDec;
   double alpha, gamma, ag2, ag3, gag, agag, totAssets, lagCapital, debt, feValue, feWgt, zValue, 
          zWgt, lkWgt, taWgt, dWgt, eqWgt, liqDec_dbl, fracRepaid, equity, totK, NKrat, intRate, 
          cash, dividend, igoods, expenses, output;
   double *tempvec;
   struct IndexInfo thisFE, thisZ, thisLK, thisTA, thisDebt, thisEquity;
           
// Timing: at the begining of each period, know total assets, total debt
// Next, decide whether to operate.  Exiting farmers "vanish"
// Finally, pick operating decisions for this period
   
   printf("Rank=%d, iSimmin=%d iSimmax=%d \n", rank, iSimmin, iSimmax);
   
   tempvec = (double *)calloc(9,sizeof(double));
// Order:  (0) liquidation decision; (1) fraction of debt repaid; (2) equity; (3) total capital; 
//         (4) owned capital; (5) dividends; (6) contractual int rate; (7) cash; (8) debt; 

   for(personInd=iSimmin; personInd<iSimmax; personInd++)    
   { 
      age0       = (int) initAges[personInd];
      year0      = (int) initYears[personInd] - 1; // GAUSS -> C++ indexing

      ftInd      = 0;
      if (ftNum > 1) { ftInd = (int) farmtypes[personInd] - 1; } // GAUSS -> C++ indexing     
      if (ftInd==0) { prodFnParms(alpha1, gamma1, &alpha, &gamma, &ag2, &ag3, &gag, &agag); } // globals
      else { prodFnParms(alpha2, gamma2, &alpha, &gamma, &ag2, &ag3, &gag, &agag); }

      feValue    = exp(feShksVec[personInd]);
      thisFE     = GetLocation(feValues, feValue, feNum);
      feInd      = thisFE.Ind1;
      feWgt      = thisFE.weight;
      if (feNum == 1) { feWgt = 1;  }
      FEIsimsMtx[0][personInd] = (float)(feInd + 1); // C++ -> GAUSS indexing
      if (feWgt<0.5) { FEIsimsMtx[0][personInd] += 1; }

      for (yearInd = 0; yearInd<year0; yearInd++)  // Skip until data appears
      {
         dividendsimsMtx[yearInd][personInd]   = -1e5;
         totKsimsMtx[yearInd][personInd]       = -1;
         NKratsimsMtx[yearInd][personInd]      = -1;
         IRsimsMtx[yearInd][personInd]         = -1;
         NWsimsMtx[yearInd][personInd]         = -1e5;
         expensesimsMtx[yearInd][personInd]    = -1;
         outputsimsMtx[yearInd][personInd]     = -1;
         cashsimsMtx[yearInd][personInd]       = -1;
         asstsimsMtx[yearInd][personInd]       = -1e5;
         debtsimsMtx[yearInd][personInd]       = -1e5;
         fracRepaidsimsMtx[yearInd+1][personInd] = -1;
         liqDecsimsMtx[yearInd][personInd]    = -1;
         agesimsMtx[yearInd][personInd]       = -1;
         zValue = exp(zShksMtx[yearInd][personInd]); // Next period's shock
         thisZ  = GetLocation(zValues, zValue, zNum); 
         zInd   = thisZ.Ind1;
         zWgt   = thisZ.weight;
         ZsimsMtx[yearInd][personInd]  = zValue; 
         ZIsimsMtx[yearInd][personInd] = (float)(zInd +1); // C++ -> GAUSS indexing
         if (zWgt<0.5) { ZIsimsMtx[yearInd+1][personInd] += 1; }                  
      }

      age        = age0;
      ageInd     = age - bornage;
//    if (ageInd>lifespan) { ageInd = lifespan; }
      if (ageInd<0) { ageInd = 0; }
      zValue     = exp(zShksMtx[year0][personInd]);
      thisZ      = GetLocation(zValues, zValue, zNum);
      zInd       = thisZ.Ind1;
      zWgt       = thisZ.weight;
      zInd2      = 0;
      if (zNum2>1) { zInd2 = zInd; }

      agesimsMtx[year0][personInd] = ((float) age);
      ZsimsMtx[year0][personInd]  = zValue; 
      ZIsimsMtx[year0][personInd] = (float)(zInd +1); // C++ -> GAUSS indexing
      if (zWgt<0.5) { ZIsimsMtx[year0][personInd] += 1; }
      
      lagCapital = initCapital[personInd];
      totAssets  = initTotAssets[personInd];
      debt       = initDebt[personInd];
      thisLK     = GetLocation(lagcapvec, lagCapital, lagcapNum);
      lkInd      = thisLK.Ind1;
      lkWgt      = thisLK.weight;
      lkInd2     = 0;
      if (lagcapNum>1) { lkInd2 = lkInd; }
      thisTA     = GetLocation(totassetvec, totAssets, totassetNum);
      taInd      = thisTA.Ind1;
      taWgt      = thisTA.weight;
      thisDebt   = GetLocation(debtvec, debt, debtNum);
      dInd       = thisDebt.Ind1;
      dWgt       = thisDebt.weight;
      liqDec     = 0;
      fracRepaid = 1;
                                             
      liqDec_dbl = intrplte7D(liqDecisionMat, ageInd, ftInd, feInd, zInd2, lkInd2, taInd, dInd,
                              dWgt, taWgt, feWgt, zWgt, lkWgt);
      fracRepaid = intrplte7D(fracRepaidMat, ageInd, ftInd, feInd, zInd2, lkInd2, taInd, dInd,
                              dWgt, taWgt, feWgt, zWgt, lkWgt);
                         
      if (liqDec_dbl>0.5) { liqDec = 1; }
      if (fracRepaid>1) { fracRepaid = 1; }
      if (fracRepaid<0) { fracRepaid = 0; }

      asstsimsMtx[year0][personInd]       = totAssets;
      debtsimsMtx[year0][personInd]       = debt;
      fracRepaidsimsMtx[year0][personInd] = fracRepaid;
      liqDecsimsMtx[year0][personInd]     = (float) liqDec;
           
      for (yearInd=year0; yearInd<(timespan+1); yearInd++)  /* calendar year */ 
      {                     
         age    = age0 + yearInd - year0;
         ageInd = age - bornage;
//       if (ageInd>lifespan) { ageInd = lifespan; }
         if (ageInd<0) { ageInd = 0; }
         agesimsMtx[yearInd][personInd] = ((float) age);
         
         if (liqDec==1)
         {
            dividendsimsMtx[yearInd][personInd]   = -1;
            totKsimsMtx[yearInd][personInd]       = -1;
            NKratsimsMtx[yearInd][personInd]      = -1;
            IRsimsMtx[yearInd][personInd]         = -1;
            NWsimsMtx[yearInd][personInd]         = -1;
            expensesimsMtx[yearInd][personInd]    = -1;
            outputsimsMtx[yearInd][personInd]     = -1;
            cashsimsMtx[yearInd][personInd]       = -1;
            if (yearInd == timespan) { continue; }

            asstsimsMtx[yearInd+1][personInd]     = -1e5;
            debtsimsMtx[yearInd+1][personInd]     = -1;
            fracRepaidsimsMtx[yearInd+1][personInd] = -1;
            liqDecsimsMtx[yearInd+1][personInd]   = 1;
            zValue = exp(zShksMtx[yearInd+1][personInd]); // Next period's shock
            thisZ  = GetLocation(zValues, zValue, zNum); 
            zInd   = thisZ.Ind1;
            zWgt   = thisZ.weight;
            ZsimsMtx[yearInd+1][personInd]  = zValue; 
            ZIsimsMtx[yearInd+1][personInd] = (float)(zInd +1); // C++ -> GAUSS indexing
            if (zWgt<0.5) { ZIsimsMtx[yearInd+1][personInd] += 1; }                  
            continue;
         }           
         
         equity     = totAssets - fracRepaid*debt;
         thisEquity = GetLocation(equityvec, equity, equityNum);
         eqInd      = thisEquity.Ind1;
         eqWgt      = thisEquity.weight;
               
//       eqInd      = LocateClosest(equityvec, equity, equityNum); // This grid is fine enough to avoid interpolation
//       eqWgt      = 1;                  

         totK       = intrplte6D(bestKFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, 
                                 eqWgt, feWgt, zWgt, lkWgt);
         NKrat      = intrplte6D(bestNKratFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, 
                                 eqWgt, feWgt, zWgt, lkWgt);
//       dividend   = intrplte6D(bestDividendFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, 
//                                 eqWgt, feWgt, zWgt, lkWgt);
         intRate    = intrplte6D(bestIntRateFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, 
                                 eqWgt, feWgt, zWgt, lkWgt);
         cash       = intrplte6D(bestCashFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, 
                                 eqWgt, feWgt, zWgt, lkWgt);
         debt       = intrplte6D(bestDebtFarm, ageInd, ftInd, feInd, zInd2, lkInd2, eqInd, 
                                 eqWgt, feWgt, zWgt, lkWgt); // This will be debt at beginning of t+1

         if (totK<0) { totK = 0; }
         if (NKrat<0) { NKrat = 0;  }
         if (intRate<bigR) { intRate = bigR; }
         if (cash<0) { cash = 0; }
         if (debt<0) { debt = 0; }

         dividend = equity + debt / intRate - totK - cash;
         if (dividend<(-c_0*eqInject)) { dividend = (-c_0*eqInject); }

         igoods   = getbaseIGoods(totK,feValue,gag,agag,ag3)*NKrat;
         expenses = igoods + fixedcost;        
         zValue   = exp(zShksMtx[yearInd+1][personInd]); // Next period's shock
         output   = feValue*zValue*getBaseRevenues(totK,igoods,gamma,ag2);  // output = revenues

         dividendsimsMtx[yearInd][personInd] = dividend;
         totKsimsMtx[yearInd][personInd]     = totK;
         NKratsimsMtx[yearInd][personInd]    = NKrat;
         NWsimsMtx[yearInd][personInd]       = equity;
         expensesimsMtx[yearInd][personInd]  = expenses;
         outputsimsMtx[yearInd][personInd]   = output;
         IRsimsMtx[yearInd][personInd]       = intRate;
         cashsimsMtx[yearInd][personInd]     = cash;

         if (yearInd==timespan) { continue; }

      // Now move to t+1 states 

         lagCapital = totK;
         totAssets  = ( (1-delta+eGK)*totK + output - expenses + cash )/bigG; // assets at t+1             
         thisTA     = GetLocation(totassetvec, totAssets, totassetNum);
         taInd      = thisTA.Ind1;
         taWgt      = thisTA.weight;              
         thisDebt   = GetLocation(debtvec, debt, debtNum);
         dInd       = thisDebt.Ind1;
         dWgt       = thisDebt.weight;          
         thisLK     = GetLocation(lagcapvec, lagCapital, lagcapNum);
         lkInd      = thisLK.Ind1;
         lkWgt      = thisLK.weight;
         lkInd2     = 0;
         if (lagcapNum>1) { lkInd2 = lkInd; }
            
         thisZ      = GetLocation(zValues, zValue, zNum);
         zInd       = thisZ.Ind1;
         zWgt       = thisZ.weight;
         zInd2      = 0;
         if (zNum2>1) { zInd2 = zInd; }
         ZsimsMtx[yearInd+1][personInd]   = zValue; 
         ZIsimsMtx[yearInd+1][personInd]  = (float)(zInd +1); // C++ -> GAUSS indexing
         if (zWgt<0.5) { ZIsimsMtx[yearInd+1][personInd] += 1; }      
         FEIsimsMtx[yearInd+1][personInd] = FEIsimsMtx[0][personInd];

         liqDec     = 0;
         fracRepaid = 1;
//       if (ageInd==lifespan) { ageInd -= 1; }

         liqDec_dbl = intrplte7D(liqDecisionMat, ageInd+1, ftInd, feInd, zInd2, lkInd2, taInd, dInd,
                                 dWgt, taWgt, feWgt, zWgt, lkWgt);
         fracRepaid = intrplte7D(fracRepaidMat, ageInd+1, ftInd, feInd, zInd2, lkInd2, taInd, dInd,
                                 dWgt, taWgt, feWgt, zWgt, lkWgt);

         if (liqDec_dbl>0.5) { liqDec = 1; } 
         if (fracRepaid>1) { fracRepaid = 1; }
         if (fracRepaid<0) { fracRepaid = 0; }
         
         asstsimsMtx[yearInd+1][personInd]       = totAssets;
         debtsimsMtx[yearInd+1][personInd]       = debt;
         fracRepaidsimsMtx[yearInd+1][personInd] = fracRepaid;
         liqDecsimsMtx[yearInd+1][personInd]     = (float) liqDec;
      } /* end loop of year */
   }  /* end loop of household */   
} //end simulation 

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/

double intrplte7D(double *******decruleMat, int ageInd, int ftInd, int feInd, int zInd2, int lkInd2, int taInd, int dInd,
                  double dWgt, double taWgt, double feWgt, double zWgt, double lkWgt)
{
   double interpVal, tempVal0, tempVal1;

   interpVal = ( (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd][dInd]*dWgt   + decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd][dInd+1]*(1-dWgt))*taWgt
               + (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd+1][dInd]*dWgt + decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][taInd+1][dInd+1]*(1-dWgt))*(1-taWgt) )*feWgt;
   if (feNum>1)
   {
      interpVal += ( (decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2][taInd][dInd]*dWgt   + decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2][taInd][dInd+1]*(1-dWgt))*taWgt
                   + (decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2][taInd+1][dInd]*dWgt + decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2][taInd+1][dInd+1]*(1-dWgt))*(1-taWgt) )*(1-feWgt);
   }

   if (zNum2>1)
   {
      tempVal0 = ( (decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2][taInd][dInd]*dWgt   + decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2][taInd][dInd+1]*(1-dWgt))*taWgt
                 + (decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2][taInd+1][dInd]*dWgt + decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2][taInd+1][dInd+1]*(1-dWgt))*(1-taWgt) )*feWgt;
      if (feNum>1)
      {
         tempVal0 += ( (decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2][taInd][dInd]*dWgt   + decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2][taInd][dInd+1]*(1-dWgt))*taWgt
                     + (decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2][taInd+1][dInd]*dWgt + decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2][taInd+1][dInd+1]*(1-dWgt))*(1-taWgt) )*(1-feWgt);
      }
      interpVal = interpVal*zWgt + tempVal0*(1-zWgt);              
   }


   if (lagcapNum>2)
   {
      tempVal1 = ( (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2+1][taInd][dInd]*dWgt   + decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2+1][taInd][dInd+1]*(1-dWgt))*taWgt
                 + (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2+1][taInd+1][dInd]*dWgt + decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2+1][taInd+1][dInd+1]*(1-dWgt))*(1-taWgt) )*feWgt;
      if (feNum>1)
      {
         tempVal1 += ( (decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2+1][taInd][dInd]*dWgt   + decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2+1][taInd][dInd+1]*(1-dWgt))*taWgt
                     + (decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2+1][taInd+1][dInd]*dWgt + decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2+1][taInd+1][dInd+1]*(1-dWgt))*(1-taWgt) )*(1-feWgt);
      }

      if (zNum2>1)
      {
         tempVal0 = ( (decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2+1][taInd][dInd]*dWgt   + decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2+1][taInd][dInd+1]*(1-dWgt))*taWgt
                    + (decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2+1][taInd+1][dInd]*dWgt + decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2+1][taInd+1][dInd+1]*(1-dWgt))*(1-taWgt) )*feWgt;
         if (feNum>1)
         {
            tempVal0 += ( (decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2+1][taInd][dInd]*dWgt   + decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2+1][taInd][dInd+1]*(1-dWgt))*taWgt
                        + (decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2+1][taInd+1][dInd]*dWgt + decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2+1][taInd+1][dInd+1]*(1-dWgt))*(1-taWgt) )*(1-feWgt);
         }
         tempVal1 = tempVal1*zWgt + tempVal0*(1-zWgt);              
      }
      interpVal = interpVal*lkWgt + tempVal1*(1-lkWgt);   
   }

   return interpVal;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/

double intrplte6D(double ******decruleMat, int ageInd, int ftInd, int feInd, int zInd2, int lkInd2, int eqInd, 
                  double eqWgt, double feWgt, double zWgt, double lkWgt)
{
   double interpVal, tempVal0, tempVal1;

   interpVal = (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][eqInd]*eqWgt + decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2][eqInd+1]*(1-eqWgt))*feWgt;
   if (feNum>1)
      interpVal += (decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2][eqInd]*eqWgt + decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2][eqInd+1]*(1-eqWgt))*(1-feWgt);

   if (zNum2>1)
   {
      tempVal0 = (decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2][eqInd]*eqWgt + decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2][eqInd+1]*(1-eqWgt))*feWgt;
      if (feNum>1)
         tempVal0 += (decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2][eqInd]*eqWgt + decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2][eqInd+1]*(1-eqWgt))*(1-feWgt);

      interpVal = interpVal*zWgt + tempVal0*(1-zWgt);              
   }

   if (lagcapNum>2)
   {
      tempVal1 = (decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2+1][eqInd]*eqWgt + decruleMat[ageInd][ftInd][feInd][zInd2][lkInd2+1][eqInd+1]*(1-eqWgt))*feWgt;
      if (feNum>1)
         tempVal1 += (decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2+1][eqInd]*eqWgt + decruleMat[ageInd][ftInd][feInd+1][zInd2][lkInd2+1][eqInd+1]*(1-eqWgt))*(1-feWgt);

      if (zNum2>1)
      {
         tempVal0 = (decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2+1][eqInd]*eqWgt + decruleMat[ageInd][ftInd][feInd][zInd2+1][lkInd2+1][eqInd+1]*(1-eqWgt))*feWgt;
         if (feNum>1)
            tempVal0 += (decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2+1][eqInd]*eqWgt + decruleMat[ageInd][ftInd][feInd+1][zInd2+1][lkInd2+1][eqInd+1]*(1-eqWgt))*(1-feWgt);

         tempVal1 = tempVal1*zWgt + tempVal0*(1-zWgt);              
      }
      interpVal = interpVal*lkWgt + tempVal1*(1-lkWgt);   
   }

   return interpVal;
}
 
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
// Write value and policy functions as column vectors to ascii files;
// Looping over indices in same order they are written so they can be read in 
// using a row-major order index function.
// Procedure split and renamed by JBJ:  01/31/12

void WriteFunctions(double **valfuncWork, double **bestCWork, double **bestNPIWork,
                    double ******valfuncFarm, double ******bestIntRateFarm, double ******bestCashFarm, 
                    double ******bestKFarm, double ******bestNKratFarm, double ******bestDividendFarm, 
                    double ******bestDebtFarm, double *******liqDecisionMat, double *******valfuncMat, 
                    double *******fracRepaidMat, double *assetvec, double *equityvec, double *lagcapvec,
                    double *debtvec, double *totassetvec,  double *feValues)
{
   FILE *valueFWP, *consumFWP, *NPIFWP, *valueFFP, *intRateFFP, *cashFFP, *totKFFP, *NKratFFP, 
        *divFFP, *debtFFP, *liqDecFP, *valueFP, *fracRepaidFP;

   int tInd, aInd, iFType, iFE, iZ, iLagK, iEquity, iTotAsset, iDebt;
   char fullpath[ADDRESS_LEN];

   valueFWP   = fopen(strcat(strcpy(fullpath, outputdir), "valueFW.txt"), "w");           /* value function, workers */
   consumFWP  = fopen(strcat(strcpy(fullpath, outputdir), "consumptionFW.txt"), "w");     /* consumption policy function */
   NPIFWP     = fopen(strcat(strcpy(fullpath, outputdir), "NPIndexWF.txt"), "w");         /* Asset index policy function */
   valueFFP   = fopen(strcat(strcpy(fullpath, outputdir), "ValueFF.txt"), "w");           /* value function, farmers */
   intRateFFP = fopen(strcat(strcpy(fullpath, outputdir), "intRateFF.txt"), "w");         /* contractual interest rate */
   cashFFP    = fopen(strcat(strcpy(fullpath, outputdir), "cashFF.txt"), "w");            /* liquid assets */
   totKFFP    = fopen(strcat(strcpy(fullpath, outputdir), "totCapitalFF.txt"), "w");      /* total capital */
   NKratFFP   = fopen(strcat(strcpy(fullpath, outputdir), "NKratioFF.txt"), "w");         /* igoods/capital, relative to optimal */
   divFFP     = fopen(strcat(strcpy(fullpath, outputdir), "dividendFF.txt"), "w");        /* dividends */
   debtFFP    = fopen(strcat(strcpy(fullpath, outputdir), "debtFF.txt"), "w");            /* debt at t+1 */
   liqDecFP   = fopen(strcat(strcpy(fullpath, outputdir), "liqDecF.txt"), "w");           /* liquidation decision */
   valueFP    = fopen(strcat(strcpy(fullpath, outputdir), "ValueF.txt"), "w");            /* value function, pre-occupation-choice */
   fracRepaidFP = fopen(strcat(strcpy(fullpath, outputdir), "fracRepaidF.txt"), "w");     /* fraction of outstanding debt repaid */

   for (aInd = 0; aInd<assetNum; aInd++)
   {
      fprintf(valueFWP, "%12.2f", assetvec[aInd]);
      fprintf(consumFWP, "%12.2f", assetvec[aInd]);
      fprintf(NPIFWP, "%12.2f", assetvec[aInd]);
         
      for (tInd = 0; tInd<lifespan; tInd++)
      {
         fprintf(valueFWP, "%22.14f", valfuncWork[tInd][aInd]);
         fprintf(consumFWP, "%12.2f", bestCWork[tInd][aInd]);
         fprintf(NPIFWP, "%8.1f", bestNPIWork[tInd][aInd]);
      }
      fprintf(valueFWP, "\n");
      fprintf(consumFWP, "\n");
      fprintf(NPIFWP, "\n");
   }

   for (iFType=0; iFType<ftNum; iFType++)
   {
      for (iFE = 0; iFE < feNum; iFE++)
      {
         for (iZ = 0; iZ < zNum2; iZ++)
         {
            for (iLagK=0; iLagK<lagcapNum; iLagK++)
            {  
               for (iEquity = 0; iEquity < equityNum; iEquity++)
               {
                  fprintf(valueFFP, "%d", iFType+1);
                  fprintf(intRateFFP, "%d", iFType+1);
                  fprintf(cashFFP, "%d", iFType+1);
                  fprintf(totKFFP, "%d", iFType+1);
                  fprintf(NKratFFP, "%d", iFType+1);
                  fprintf(divFFP, "%d", iFType+1);
                  fprintf(debtFFP, "%d", iFType+1);

                  fprintf(valueFFP, "%10.6f", feValues[iFE]);
                  fprintf(intRateFFP, "%10.6f", feValues[iFE]);
                  fprintf(cashFFP, "%10.6f", feValues[iFE]);
                  fprintf(totKFFP, "%10.6f", feValues[iFE]);
                  fprintf(NKratFFP, "%10.6f", feValues[iFE]);
                  fprintf(divFFP, "%10.6f", feValues[iFE]);
                  fprintf(debtFFP, "%10.6f", feValues[iFE]);

                  fprintf(valueFFP, "%12.2f", lagcapvec[iLagK]);
                  fprintf(intRateFFP, "%12.2f", lagcapvec[iLagK]);
                  fprintf(cashFFP, "%12.2f", lagcapvec[iLagK]);
                  fprintf(totKFFP, "%12.2f", lagcapvec[iLagK]);
                  fprintf(NKratFFP, "%12.2f", lagcapvec[iLagK]);
                  fprintf(divFFP, "%12.2f", lagcapvec[iLagK]);
                  fprintf(debtFFP, "%12.2f", lagcapvec[iLagK]);

                  fprintf(valueFFP, "%12.2f", equityvec[iEquity]);
                  fprintf(intRateFFP, "%12.2f", equityvec[iEquity]);
                  fprintf(cashFFP, "%12.2f", equityvec[iEquity]);
                  fprintf(totKFFP, "%12.2f", equityvec[iEquity]);
                  fprintf(NKratFFP, "%12.2f", equityvec[iEquity]);
                  fprintf(divFFP, "%12.2f", equityvec[iEquity]);
                  fprintf(debtFFP, "%12.2f", equityvec[iEquity]);
            
                  for (tInd = 0; tInd<lifespan; tInd++)
                  {
                     if ((valfuncFarm[tInd][iFType][iFE][iZ][iLagK][iEquity]*valfuncFarm[tInd][iFType][iFE][iZ][iLagK][iEquity])>10000000000)
                       { fprintf(valueFFP, "%22.2f", valfuncFarm[tInd][iFType][iFE][iZ][iLagK][iEquity]); }
                     else
                       { fprintf(valueFFP, "%22.14f", valfuncFarm[tInd][iFType][iFE][iZ][iLagK][iEquity]); }
                     fprintf(intRateFFP, "%12.7f", bestIntRateFarm[tInd][iFType][iFE][iZ][iLagK][iEquity]);
                     fprintf(cashFFP, "%12.2f", bestCashFarm[tInd][iFType][iFE][iZ][iLagK][iEquity]);
                     fprintf(totKFFP, "%12.2f", bestKFarm[tInd][iFType][iFE][iZ][iLagK][iEquity]);
                     fprintf(NKratFFP, "%6.3f", bestNKratFarm[tInd][iFType][iFE][iZ][iLagK][iEquity]);
                     fprintf(divFFP, "%12.2f", bestDividendFarm[tInd][iFType][iFE][iZ][iLagK][iEquity]);
                     fprintf(debtFFP, "%12.2f", bestDebtFarm[tInd][iFType][iFE][iZ][iLagK][iEquity]);
                  }
                  fprintf(valueFFP, "\n");
                  fprintf(intRateFFP, "\n");
                  fprintf(cashFFP, "\n");
                  fprintf(totKFFP, "\n");
                  fprintf(NKratFFP, "\n");
                  fprintf(divFFP, "\n");
                  fprintf(debtFFP, "\n");
               }


               for (iTotAsset = 0; iTotAsset<totassetNum; iTotAsset++)
               {
                  for (iDebt = 0; iDebt<debtNum; iDebt++)
                  {
                     fprintf(valueFP, "%d", iFType+1);
                     fprintf(liqDecFP, "%d", iFType+1);
                     fprintf(fracRepaidFP, "%d", iFType+1);
                     fprintf(valueFP, "%10.6f", feValues[iFE]);
                     fprintf(liqDecFP, "%10.6f", feValues[iFE]);
                     fprintf(fracRepaidFP, "%10.6f", feValues[iFE]);
                     fprintf(valueFP, "%12.2f", lagcapvec[iLagK]);
                     fprintf(liqDecFP, "%12.2f", lagcapvec[iLagK]);
                     fprintf(fracRepaidFP, "%12.2f", lagcapvec[iLagK]);
                     fprintf(valueFP, "%12.2f", totassetvec[iTotAsset]);
                     fprintf(liqDecFP, "%12.2f", totassetvec[iTotAsset]);
                     fprintf(fracRepaidFP, "%12.2f", totassetvec[iTotAsset]);
                     fprintf(valueFP, "%12.2f", debtvec[iDebt]);
                     fprintf(liqDecFP, "%12.2f", debtvec[iDebt]);
                     fprintf(fracRepaidFP, "%12.2f", debtvec[iDebt]);
            
                     for (tInd = 0; tInd<lifespan+1; tInd++)
                     {
                        if ((valfuncMat[tInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt]*valfuncMat[tInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt])>10000000000)
                           { fprintf(valueFP, "%22.2f", valfuncMat[tInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt]); }
                        else
                           { fprintf(valueFP, "%22.14f", valfuncMat[tInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt]); }
                        fprintf(liqDecFP, "%3d", (int) liqDecisionMat[tInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt]);
                        fprintf(fracRepaidFP, "%12.8f", fracRepaidMat[tInd][iFType][iFE][iZ][iLagK][iTotAsset][iDebt]);
                     }

                     fprintf(liqDecFP, "\n");
                     fprintf(valueFP, "\n");
                     fprintf(fracRepaidFP, "\n");
                  }
               }
            }
         }
      }
   }

// End writing to files 
   fclose(valueFWP);
   fclose(consumFWP);
   fclose(NPIFWP);
   fclose(valueFFP);
   fclose(intRateFFP);
   fclose(cashFFP);
   fclose(totKFFP);
   fclose(NKratFFP);
   fclose(divFFP);
   fclose(debtFFP);
   fclose(liqDecFP);
   fclose(valueFP);
   fclose(fracRepaidFP);
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
// Write simulation results to ascii files;
// Rows correspond to farms, columns to years 

void WriteSims(double **FEIsimsMtx, double **ZsimsMtx, double **ZIsimsMtx, double **asstsimsMtx, 
               double **dividendsimsMtx, double **totKsimsMtx, double **NKratsimsMtx, double **cashsimsMtx,  
               double **IRsimsMtx, double **debtsimsMtx, double **NWsimsMtx, double **fracRepaidsimsMtx, 
               double **outputsimsMtx, double **liqDecsimsMtx, double **agesimsMtx, double **expensesimsMtx)
{
   FILE *FEindxSP, *ZvalsSP, *ZindxSP, *assetsSP, *divsSP, *totKSP, *NKratSP, *cashSP, *intRSP, *debtSP, *equitySP, *fracRPSP,
        *outputSP, *liqDecSP, *ageSP, *expenseSP;

   int iPerson, iYear;
   char fullpath[ADDRESS_LEN];

   FEindxSP   = fopen(strcat(strcpy(fullpath, outputdir), "FEindxS.txt"), "w");          /* Index numbers, fixed Effect TFP shock */
   ZvalsSP    = fopen(strcat(strcpy(fullpath, outputdir), "ZValsS.txt"), "w");           /* Simulated values, transitory TFP shock */
   ZindxSP    = fopen(strcat(strcpy(fullpath, outputdir), "ZindxS.txt"), "w");           /* Index numbers, TFP shock */
   assetsSP   = fopen(strcat(strcpy(fullpath, outputdir), "assetsS.txt"), "w");          /* Total Assets period t */
   divsSP     = fopen(strcat(strcpy(fullpath, outputdir), "divsS.txt"), "w");            /* Dividends/consumption */
   totKSP     = fopen(strcat(strcpy(fullpath, outputdir), "totKS.txt"), "w");            /* Total capital */
   NKratSP    = fopen(strcat(strcpy(fullpath, outputdir), "NKratioS.txt"), "w");         /* igoods/capital ratio */
   cashSP     = fopen(strcat(strcpy(fullpath, outputdir), "cashS.txt"), "w");            /* Cash/liquid assets */
   intRSP     = fopen(strcat(strcpy(fullpath, outputdir), "intRateS.txt"), "w");         /* Contractual interest rates */
   debtSP     = fopen(strcat(strcpy(fullpath, outputdir), "debtS.txt"), "w");            /* Debt for period t+1 */
   equitySP   = fopen(strcat(strcpy(fullpath, outputdir), "equityS.txt"), "w");          /* Net worth for period t */
   fracRPSP   = fopen(strcat(strcpy(fullpath, outputdir), "fracRPS.txt"), "w");          /* Fraction of outstanding debt repaid */
   outputSP   = fopen(strcat(strcpy(fullpath, outputdir), "outputS.txt"), "w");          /* Output/revenues */
   liqDecSP   = fopen(strcat(strcpy(fullpath, outputdir), "liqDecS.txt"), "w");          /* Liquidation decisions */
   ageSP      = fopen(strcat(strcpy(fullpath, outputdir), "ageS.txt"), "w");             /* Age of farm head */
   expenseSP  = fopen(strcat(strcpy(fullpath, outputdir), "expenseS.txt"), "w");         /* Operating expenditures */

   for (iPerson = 0; iPerson<numSims; iPerson++)
   {
      for (iYear = 0; iYear<(timespan+1); iYear++)
      {
         fprintf(FEindxSP, "%4.0f", FEIsimsMtx[iYear][iPerson]);
         fprintf(ZvalsSP, "%12.8f", ZsimsMtx[iYear][iPerson]);
         fprintf(ZindxSP, "%4.0f", ZIsimsMtx[iYear][iPerson]);
         fprintf(assetsSP, "%12.3f", asstsimsMtx[iYear][iPerson]);
         fprintf(divsSP, "%12.3f", dividendsimsMtx[iYear][iPerson]);
         fprintf(totKSP, "%12.3f", totKsimsMtx[iYear][iPerson]);
         fprintf(NKratSP, "%12.3f", NKratsimsMtx[iYear][iPerson]);
         fprintf(cashSP, "%12.3f", cashsimsMtx[iYear][iPerson]);
         fprintf(intRSP, "%11.7f", IRsimsMtx[iYear][iPerson]);
         fprintf(debtSP, "%12.3f", debtsimsMtx[iYear][iPerson]);
         fprintf(equitySP, "%12.3f", NWsimsMtx[iYear][iPerson]);
         fprintf(fracRPSP, "%10.6f", fracRepaidsimsMtx[iYear][iPerson]);
         fprintf(outputSP, "%12.3f", outputsimsMtx[iYear][iPerson]);
         fprintf(liqDecSP, "%4.0f", liqDecsimsMtx[iYear][iPerson]);
         fprintf(ageSP, "%5.0f", agesimsMtx[iYear][iPerson]);
         fprintf(expenseSP, "%12.3f", expensesimsMtx[iYear][iPerson]);
      }

      fprintf(FEindxSP, "\n");
      fprintf(ZvalsSP, "\n");
      fprintf(ZindxSP, "\n");
      fprintf(assetsSP, "\n");
      fprintf(divsSP, "\n");
      fprintf(totKSP, "\n");
      fprintf(NKratSP, "\n");
      fprintf(cashSP, "\n");
      fprintf(intRSP, "\n");
      fprintf(debtSP, "\n");
      fprintf(equitySP, "\n");
      fprintf(fracRPSP, "\n");
      fprintf(outputSP, "\n");
      fprintf(liqDecSP, "\n");
      fprintf(ageSP, "\n");
      fprintf(expenseSP, "\n");
   }

// End writing to files 
   fclose(FEindxSP);
   fclose(ZvalsSP);
   fclose(ZindxSP);
   fclose(assetsSP);
   fclose(divsSP);
   fclose(totKSP);
   fclose(NKratSP);
   fclose(cashSP);
   fclose(intRSP);
   fclose(debtSP);
   fclose(equitySP);
   fclose(fracRPSP);
   fclose(outputSP);
   fclose(liqDecSP);
   fclose(ageSP);
   fclose(expenseSP);
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  SETUP2DMAT:  Initialize 2-dimensional matrices 
                 Each row of dataMat is a subsection of dataVec    
                 This economizes on memory and reduces copying  
*/
double **SetUp2DMat(double *dataVec, int numRows, int numCols)
{
   double **dataMat;
   int iRow;

   dataMat = (double **)malloc(numRows*sizeof(double *));
   for(iRow=0; iRow<numRows; iRow++)
      dataMat[iRow] = &dataVec[iRow*numCols];

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  SETUP3DMAT:  Initialize 3-dimensional matrices 
                 Each row of dataMat is a subsection of dataVec    
                 This economizes on memory and reduces copying  
*/
double ***SetUp3DMat(double *dataVec, int numDim1, int numRows, int numCols)
{
   double ***dataMat;
   int iDim1, iRow, i;

   dataMat = (double ***)malloc(numDim1*sizeof(double **));
   for(iDim1=0; iDim1<numDim1; iDim1++)
   {
      dataMat[iDim1] = (double **)malloc(numRows*sizeof(double *));
      for(iRow=0; iRow<numRows; iRow++)
      {
         i = iDim1*(numRows*numCols) + iRow*numCols;
         dataMat[iDim1][iRow] = &dataVec[i];
      }
   }
   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  SETUP4DMAT:  Initialize 4-dimensional matrices 
                 Each row of dataMat is a subsection of dataVec    
                 This economizes on memory and reduces copying  
*/
double ****SetUp4DMat(double *dataVec, int numDim1, int numDim2, int numRows, int numCols)
{
   double ****dataMat;
   int iDim1, iDim2, iRow, i;

   dataMat = (double ****)malloc(numDim1*sizeof(double ***));
   for(iDim1=0; iDim1<numDim1; iDim1++)
   {
      dataMat[iDim1] = (double ***)malloc(numDim2*sizeof(double **));
      for(iDim2=0; iDim2<numDim2; iDim2++)
      {
         dataMat[iDim1][iDim2] = (double **)malloc(numRows*sizeof(double *));
         for(iRow=0; iRow<numRows; iRow++)
         {
            i = iDim1*(numDim2*numRows*numCols) + iDim2*(numRows*numCols) + iRow*numCols;
            dataMat[iDim1][iDim2][iRow] = &dataVec[i];
         }
      }
   }
   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT2D:  Initialize 2-dimensional matrix and fill it with zeros            */

double **ZeroMat2D(int numRows, int numCols)
{
   double **dataMat;
   int iRow;

   dataMat = (double **)malloc(numRows*sizeof(double *));
   for(iRow=0; iRow<numRows; iRow++)
      dataMat[iRow] = (double *)calloc(numCols,sizeof(double));

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT2DI:  Initialize 2-dimensional matrix and fill it with integer zeros   */

int **ZeroMat2DI(int numRows, int numCols)
{
   int **dataMat;
   int iRow;

   dataMat = (int **)malloc(numRows*sizeof(int *));
   for(iRow=0; iRow<numRows; iRow++)
      dataMat[iRow] = (int *)calloc(numCols,sizeof(int));

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT3D:  Initialize 3-dimensional matrix and fill it with zeros            */

double ***ZeroMat3D(int numDim1, int numRows, int numCols)
{
   double ***dataMat;
   int iDim1;

   dataMat = (double ***)malloc(numDim1*sizeof(double **));
   for(iDim1=0; iDim1<numDim1; iDim1++)
      dataMat[iDim1] = ZeroMat2D(numRows, numCols);

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT3DI:  Initialize 3-dimensional matrix and fill it with integer zeros   */

int ***ZeroMat3DI(int numDim1, int numRows, int numCols)
{
   int ***dataMat;
   int iDim1;

   dataMat = (int ***)malloc(numDim1*sizeof(int **));
   for(iDim1=0; iDim1<numDim1; iDim1++)
      dataMat[iDim1] = ZeroMat2DI(numRows, numCols);
 
   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT4D:  Initialize 4-dimensional matrix and fill it with zeros            */

double ****ZeroMat4D(int numDim1, int numDim2, int numRows, int numCols)
{
   double ****dataMat;
   int iDim1;

   dataMat = (double ****)malloc(numDim1*sizeof(double ***));
   for(iDim1=0; iDim1<numDim1; iDim1++)
      dataMat[iDim1] = ZeroMat3D(numDim2, numRows, numCols);

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT4DI:  Initialize 4-dimensional matrix and fill it with integer zeros   */

int ****ZeroMat4DI(int numDim1, int numDim2, int numRows, int numCols)
{
   int ****dataMat;
   int iDim1;

   dataMat = (int ****)malloc(numDim1*sizeof(int ***));
   for(iDim1=0; iDim1<numDim1; iDim1++)
      dataMat[iDim1] = ZeroMat3DI(numDim2, numRows, numCols);

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT5D:  Initialize 5-dimensional matrix and fill it with zeros            */

double *****ZeroMat5D(int numDim1, int numDim2, int numDim3, int numRows, int numCols)
{
   double *****dataMat;
   int iDim1;

   dataMat = (double *****)malloc(numDim1*sizeof(double ****));
   for (iDim1 = 0; iDim1 < numDim1; iDim1++)
      dataMat[iDim1] = ZeroMat4D(numDim2, numDim3, numRows, numCols);

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT5DI:  Initialize 5-dimensional matrix and fill it with integer zeros   */

int *****ZeroMat5DI(int numDim1, int numDim2, int numDim3, int numRows, int numCols)
{
   int *****dataMat;
   int iDim1;

   dataMat = (int *****)malloc(numDim1*sizeof(int ****));
   for(iDim1=0; iDim1<numDim1; iDim1++)
      dataMat[iDim1] = ZeroMat4DI(numDim2, numDim3, numRows, numCols);

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT6D:  Initialize 6-dimensional matrix and fill it with zeros            */

double ******ZeroMat6D(int numDim1, int numDim2, int numDim3, int numDim4, 
                       int numRows, int numCols)
{
   double ******dataMat;
   int iDim1;

   dataMat = (double ******)malloc(numDim1*sizeof(double *****));
   for (iDim1 = 0; iDim1 < numDim1; iDim1++)
      dataMat[iDim1] = ZeroMat5D(numDim2, numDim3, numDim4, numRows, numCols);

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT6DI:  Initialize 6-dimensional matrix and fill it with integer zeros   */

int ******ZeroMat6DI(int numDim1, int numDim2, int numDim3, int numDim4, 
                     int numRows, int numCols)
{
   int ******dataMat;
   int iDim1;

   dataMat = (int ******)malloc(numDim1*sizeof(int *****));
   for(iDim1=0; iDim1<numDim1; iDim1++)
      dataMat[iDim1] = ZeroMat5DI(numDim2, numDim3, numDim4, numRows, numCols);

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT7D:  Initialize 6-dimensional matrix and fill it with zeros            */

double *******ZeroMat7D(int numDim1, int numDim2, int numDim3, int numDim4, 
                        int numDim5, int numRows, int numCols)
{
   double *******dataMat;
   int iDim1;

   dataMat = (double *******)malloc(numDim1*sizeof(double ******));
   for (iDim1 = 0; iDim1 < numDim1; iDim1++)
      dataMat[iDim1] = ZeroMat6D(numDim2, numDim3, numDim4, numDim5, numRows, numCols);

   return dataMat;
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROMAT7DI:  Initialize 7-dimensional matrix and fill it with integer zeros   */

int *******ZeroMat7DI(int numDim1, int numDim2, int numDim3, int numDim4, 
                      int numDim5, int numRows, int numCols)
{
   int *******dataMat;
   int iDim1;

   dataMat = (int *******)malloc(numDim1*sizeof(int ******));
   for(iDim1=0; iDim1<numDim1; iDim1++)
      dataMat[iDim1] = ZeroMat6DI(numDim2, numDim3, numDim4, numDim5, numRows, numCols);

   return dataMat;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT2D:  Frees memory dynamically allocated to a 2-dimensional matrix      */

void FreeMat2D(double **dataMat, int numRows, int numCols)
{
   int iRow;

   for(iRow=0; iRow<numRows; iRow++)
      free (dataMat[iRow]);

   free(dataMat);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT2DI:  Frees memory dynamically allocated to a 2-dimensional matrix     */

void FreeMat2DI(int **dataMat, int numRows, int numCols)
{
   int iRow;

   for (iRow = 0; iRow<numRows; iRow++)
      free(dataMat[iRow]);

   free(dataMat);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT3D:  Frees memory dynamically allocated to a 3-dimensional matrix      */

void FreeMat3D(double ***dataMat, int numDim1, int numRows, int numCols)
{
   int iDim1;

   for(iDim1=0; iDim1<numDim1; iDim1++)
      FreeMat2D(dataMat[iDim1], numRows, numCols);

   free(dataMat);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT3DI:  Frees memory dynamically allocated to a 3-dimensional matrix     */

void FreeMat3DI(int ***dataMat, int numDim1, int numRows, int numCols)
{
   int iDim1;

   for (iDim1 = 0; iDim1<numDim1; iDim1++)
      FreeMat2DI(dataMat[iDim1], numRows, numCols);

   free(dataMat);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT4D:  Frees memory dynamically allocated to a 4-dimensional matrix      */

void FreeMat4D(double ****dataMat, int numDim1, int numDim2, int numRows, int numCols)
{
   int iDim1;

   for(iDim1=0; iDim1<numDim1; iDim1++)
      FreeMat3D(dataMat[iDim1], numDim2, numRows, numCols);

   free(dataMat);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT4DI:  Frees memory dynamically allocated to a 4-dimensional matrix     */

void FreeMat4DI(int ****dataMat, int numDim1, int numDim2, int numRows, int numCols)
{
   int iDim1;

   for (iDim1 = 0; iDim1<numDim1; iDim1++)
      FreeMat3DI(dataMat[iDim1], numDim2, numRows, numCols);

   free(dataMat);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT5D:  Frees memory dynamically allocated to a 5-dimensional matrix      */

void FreeMat5D(double *****dataMat, int numDim1, int numDim2, int numDim3, int numRows, int numCols)
{
   int iDim1;

   for(iDim1=0; iDim1<numDim1; iDim1++)
      FreeMat4D(dataMat[iDim1], numDim2, numDim3, numRows, numCols);

   free(dataMat);
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT5DI:  Frees memory dynamically allocated to a 5-dimensional matrix     */

void FreeMat5DI(int *****dataMat, int numDim1, int numDim2, int numDim3, int numRows, int numCols)
{
   int iDim1;

   for (iDim1 = 0; iDim1<numDim1; iDim1++)
      FreeMat4DI(dataMat[iDim1], numDim2, numDim3, numRows, numCols);

   free(dataMat);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT6D:  Frees memory dynamically allocated to a 6-dimensional matrix      */

void FreeMat6D(double ******dataMat, int numDim1, int numDim2, int numDim3, int numDim4, 
               int numRows, int numCols)
{
   int iDim1;

   for(iDim1=0; iDim1<numDim1; iDim1++)
      FreeMat5D(dataMat[iDim1], numDim2, numDim3, numDim4, numRows, numCols);

   free(dataMat);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT6DI:  Frees memory dynamically allocated to a 6-dimensional matrix     */

void FreeMat6DI(int ******dataMat, int numDim1, int numDim2, int numDim3, int numDim4, 
                int numRows, int numCols)
{
   int iDim1;

   for (iDim1 = 0; iDim1<numDim1; iDim1++)
      FreeMat5DI(dataMat[iDim1], numDim2, numDim3, numDim4, numRows, numCols);

   free(dataMat);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT7D:  Frees memory dynamically allocated to a 7-dimensional matrix      */

void FreeMat7D(double *******dataMat, int numDim1, int numDim2, int numDim3, int numDim4, 
               int numDim5, int numRows, int numCols)
{
   int iDim1;

   for(iDim1=0; iDim1<numDim1; iDim1++)
      FreeMat6D(dataMat[iDim1], numDim2, numDim3, numDim4, numDim5, numRows, numCols);

   free(dataMat);
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  FREEMAT7DI:  Frees memory dynamically allocated to a 7-dimensional matrix     */

void FreeMat7DI(int *******dataMat, int numDim1, int numDim2, int numDim3, int numDim4, 
                int numDim5, int numRows, int numCols)
{
   int iDim1;

   for (iDim1 = 0; iDim1<numDim1; iDim1++)
      FreeMat6DI(dataMat[iDim1], numDim2, numDim3, numDim4, numDim5, numRows, numCols);

   free(dataMat);
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  Below is C/GAUSS I/O code written by Ken Housinger                       */

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  ZEROGMAT:  Initializes matrix structure and fills it with zeros  */

GMatrix zeroGMat(unsigned int recsize)
{
   GMatrix mat = {1, 1};
   mat.m = recsize;
   mat.n = 1;
   mat.data = (double *)calloc(recsize,sizeof(double));
   return mat;
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
/*  This function takes a char* and either reverses bits within bytes or bytes */
/*  themselves or both.  It returns a pointer to the original buffer which is  */
/*  altered. */


unsigned char * gread(unsigned char *inbuf, int bytes, int byte_rev, int bit_rev) 
   {
    unsigned char *tempbuf;
    unsigned char tempbyte, tempbit;
    int i, j;

    tempbuf = (unsigned char *) malloc(bytes);
    for (i = 0; i < bytes; i++) 
       {
        if (byte_rev) *(tempbuf + i) = *(inbuf + bytes - i - 1);
        else *(tempbuf + i) = *(inbuf + i);

        if (bit_rev) 
           {
            tempbyte = 0;
            for (j = 0; j < CHAR_BIT; j++) 
               {
                tempbit = *(tempbuf + i) >> (CHAR_BIT - j - 1);
                tempbit = tempbit << (CHAR_BIT - 1);
                tempbit = tempbit >> (CHAR_BIT - j - 1);
                tempbyte = tempbyte | tempbit;
               }
            *(tempbuf + i) = tempbyte;
           }
       }

    for (i = 0; i < bytes; i++)
        *(inbuf + i) = *(tempbuf + i);
    free(tempbuf);

    return(inbuf);
   }

//*--------------------------------------------------------------------------------*/
//*--------------------------------------------------------------------------------*/
//*  This function reads a Gauss v5.0 fmt file into a Matrix structure and  */
//*  returns the structure.  */

GMatrix gau5read(char *fmt) 
   {
//*  Initialize the matrix to be 1x1 and the byte/bit order to 0.  */
    GMatrix mat = {1, 1}; 
    unsigned int i;
    int type, byte, bit;
    unsigned char *cread;
    int bit_rev = 0, byte_rev = 0;
    FILE *fd;

    if (sizeof(int) != 4 || sizeof(double) != 8 || CHAR_BIT != 8) 
       {
        printf("Incompatable machine architecture.\n");
        return (mat);
       }

//*  Allocate enough space to store the header.  */
    cread = (unsigned char *) malloc(BASIC_HEADER_LEN); 
//*  Open *fmt for reading only.  */
    fd = fopen(fmt, "rb"); 
  
//*  Read the basic header (128 bytes) all at once.  */

    fread((void *) cread, 1, BASIC_HEADER_LEN, fd);
    byte = (int) *(cread + (BYTE_POS * sizeof(int)));  /* (0=Backward) */
    bit = (int) *(cread + (BIT_POS * sizeof(int)));    /* (0=Backward) */

//*  To get some system independence, we detect whether we have to reverse */
//*  the bytes or bits or both.  If x and x_SYSTEM match, no reverse is  */
//*  necessary. */

    if ((bit || BIT_SYSTEM) && !(bit && BIT_SYSTEM)) bit_rev=1;
    if ((byte || BYTE_SYSTEM) && !(byte && BYTE_SYSTEM)) byte_rev=1;

    type = *( (int *) gread((cread + (TYPE_POS * sizeof(int))), sizeof(int), 
                             byte_rev, bit_rev) );

//*  If the fmt file type is not a scalar, there are another two */
//*  ints of header giving the values of m and n.  If a matrix, also reset n. */

    if (type > SCALAR) 
       { 
        fread((void *) cread, 1, sizeof(int), fd);
        mat.m = *((unsigned int *) gread(cread, sizeof(int), byte_rev, bit_rev));
        fread((void *) cread, 1, sizeof(int), fd);
        if (type == MATRIX)
          mat.n = *((unsigned int *) gread(cread, sizeof(int), byte_rev, bit_rev));
       } 

//*  Allocate memory for the matrix.  The amount needed is m * n * sizeof(double). */
//*  Next, read in the data all at once.  Then use gread to reverse */
//*  bytes/bits if necessary. */

    free(cread);

    mat.data = (double *) malloc(mat.m * mat.n * sizeof(double));
    fread((void *) mat.data, sizeof(double), mat.m * mat.n, fd);
    if (byte_rev || bit_rev)
      for(i = 0; i < mat.m * mat.n; i++)
        gread((unsigned char *) mat.data + (i * sizeof(double)), sizeof(double), 
               byte_rev, bit_rev);

    fclose(fd);

    return (mat);
   }

//*--------------------------------------------------------------------------------*/
//*--------------------------------------------------------------------------------*/
//*  This function writes a Gauss v5.0 fmt file from a Matrix structure. */

void gau5write(char *fmt, GMatrix mat) 
{
//*  This ugly mess is the basic header. */

    unsigned int header[(BASIC_HEADER_LEN / sizeof(int)) + 2] = 
        {0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0,
         0xabcdef01,1, 0, 1, 1008, sizeof(double), 0, 1,
         SCALAR, 1, 0, BASIC_HEADER_LEN};

    FILE *fd;

    if (sizeof(int) != 4 || sizeof(double) != 8 || CHAR_BIT != 8) 
       {
        printf("Incompatible machine architecture.\n");
        return;
       }

//*  If forward byte, make 6th int 0xffffffff. */
//*  If forward bit, make 7th int 0xffffffff. */

    if (BYTE_SYSTEM) header[BYTE_POS] = 0xffffffff;
    if (BIT_SYSTEM) header[BIT_POS] = 0xffffffff;

//*  If not a scalar, increase the 16th int by 1 and the 19th int (header */ 
//*  length) by 8 (2 * sizeof(int)).  Also, set m in int 33. */

    if (!(mat.m * mat.n == 1)) 
       {
        header[TYPE_POS] += 1;
        header[HEADER_LEN_POS] += (2 * sizeof(int));
        header[M_POS] = mat.m;

    /*  If not a vector (and not a scalar), increase the 16th int by 1 again */
    /*  and set m in int 34. */
        if (!(mat.n == 1)) 
           {
            header[TYPE_POS] += 1;
            header[N_POS] = mat.n;
           }
       }
  /*
  **Open fmt for writing and create if it does not exist.  If you create it,
  **make it a regular file with permissions 0640.  See comment in gau5read
  **for detail on how read (and similarly, write) work.
  **
  **Order: Write the basic header
  **       If not a scalar, write the other 2 ints of header 
  **       Write the m * n elements of data
  */

    fd = fopen(fmt, "wb"); 
    if ((mat.m * mat.n == 1))
      fwrite((void *) header, 1, BASIC_HEADER_LEN, fd);
    else
      fwrite((void *) header, 1, BASIC_HEADER_LEN + (2 * sizeof(int)), fd);
    fwrite((void *) mat.data, sizeof(double), mat.m * mat.n, fd);
    fclose(fd);
   }  
//*--------------------------------------------------------------------------------*/
//*--------------------------------------------------------------------------------*/

// Hjaltes nye funktion
GMatrix readDoubleList(const char* filename) {
  FILE* fp = fopen(filename, "r"); // Open the file for reading

  if (fp == NULL) {
    return (GMatrix){0}; // Return an empty GMatrix on error
  }

  // Count the number of elements (lines) in the file (optional)
  int numElements = 0;
  char buffer[100];
  while (fgets(buffer, sizeof(buffer), fp) != NULL) {
    numElements++;
  }
  rewind(fp); // Rewind the file pointer to the beginning

  // Create a GMatrix with numElements rows and 1 column
  GMatrix mat = zeroGMat(numElements);

  // Read doubles from the file and store them in the matrix
  for (int i = 0; i < numElements; i++) {
    if (fscanf(fp, "%lf", &mat.data[i]) != 1) {
      fprintf(stderr, "Error reading double at line %d\n", i + 1);
      free(mat.data); // Free memory on error
      return (GMatrix){0};
    }
  }

  fclose(fp); // Close the file
  return mat;
}

void writeGMatrixToFile(const char* filename, const GMatrix* mat) {
  FILE* fp = fopen(filename, "w"); // Open the file for writing

  if (fp == NULL) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    return;
  }

  // Write each element of the matrix
  for (int i = 0; i < mat->n; i++) {
    for (int j = 0; j < mat->m; j++) {
      fprintf(fp, "%lf\n", mat->data[i * mat->m + j]); // Print element and space
    }
    fprintf(fp, "\n"); // Newline after each row
  }

  fclose(fp); // Close the file
}
