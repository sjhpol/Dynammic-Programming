// Little script that uses Jones&Pratap code to convert annoying GAUSS format to a nice, god-fearing, .txt-format

// Deals with GAUSS->GAUSS stuff that lives in IOFILES. But since we're calling Python->Python, 
// we don't need to call this in the inner loop. So it lives separately. 

// We don't -fopenmp here, bc I've had enough of that lol

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <limits.h>

//#include <omp.h> // Open MP

#define ADDRESS_LEN 300
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

typedef struct {   /* this defines a matrix with given # of rows and */
  unsigned int m;  /* columns and gives the address of the first element */
  unsigned int n;
  double *data;} GMatrix;

//  GAUSS-C++ I/O programs written by K. Housinger 
unsigned char * gread(unsigned char *inbuf, int bytes, int byte_reverse, int bit_reverse);
GMatrix gau5read(char *fmt);
GMatrix zeroGMat(unsigned int recsize);
void gau5write(char *fmt, GMatrix mat); /* reads vector from hard drive*/

// Hjalte's new functions
GMatrix readDoubleList(const char* filename);
void writeGMatrixToFile(const char* filename, const GMatrix* mat);

char rootdir[ADDRESS_LEN]; 
char outputdir[ADDRESS_LEN]; 
char datadir[ADDRESS_LEN]; 
char fullpath[ADDRESS_LEN];

int main(int argc, char *argv[])
{
    int switchMac = 1;

    if (switchMac==1)  // Hjalte
    {
        strcpy(rootdir,"/Users/hjaltewallin/Code/Jones_Pratap_AER_2017-0370_Archive/estimation_fake/iofiles/");
        strcpy(outputdir, "/Users/hjaltewallin/Code/DP-MESTERNE/Dynammic-Programming/iofiles/"); 
        strcpy(datadir, "/Users/hjaltewallin/Code/DP-MESTERNE/Dynammic-Programming/data/"); 
    }
    else if (switchMac==0) 
    {
        strcpy(rootdir, "c:\\users\\sangeeta\\dropbox\\farms\\estimation_fake\\iofiles\\");
        strcpy(outputdir, "c:\\users\\sangeeta\\dropbox\\farms\\estimation_fake\\output\\");
        strcpy(datadir, "c:\\users\\sangeeta\\dropbox\\farms\\estimation_fake\\data\\");
    }

    //obsSim, iobsSim, dvgobsSim, FEshks, IDsim, simwgts, ftype_sim
    GMatrix obsSimPtr, iobsSimPtr, dvgobsSimPtr, FEshksPtr, IDsimPtr, simwgtsPtr, ftype_simPtr;

    // Read in .fmt files
    obsSimPtr       = gau5read(strcat(strcpy(fullpath,rootdir),"obssim.fmt")); 
    iobsSimPtr      = gau5read(strcat(strcpy(fullpath,rootdir),"iobsSim.fmt")); 
    dvgobsSimPtr    = gau5read(strcat(strcpy(fullpath,rootdir),"dvgobsSim.fmt")); 
    FEshksPtr       = gau5read(strcat(strcpy(fullpath,rootdir),"FEshks.fmt")); 
    IDsimPtr        = gau5read(strcat(strcpy(fullpath,rootdir),"IDsim.fmt"));
    simwgtsPtr      = gau5read(strcat(strcpy(fullpath,rootdir),"simwgts.fmt"));
    ftype_simPtr    = gau5read(strcat(strcpy(fullpath,rootdir),"ftype_sim.fmt"));

    // Write out .txt files
    writeGMatrixToFile(strcat(strcpy(fullpath, outputdir), "obssim.txt"), &obsSimPtr);
    writeGMatrixToFile(strcat(strcpy(fullpath, outputdir), "iobsSim.txt"), &iobsSimPtr);
    writeGMatrixToFile(strcat(strcpy(fullpath, outputdir), "dvgobsSim.txt"), &dvgobsSimPtr);
    writeGMatrixToFile(strcat(strcpy(fullpath, outputdir), "FEshks.txt"), &FEshksPtr);
    writeGMatrixToFile(strcat(strcpy(fullpath, outputdir), "IDsim.txt"), &IDsimPtr);
    writeGMatrixToFile(strcat(strcpy(fullpath, outputdir), "simwgts.txt"), &simwgtsPtr);
    writeGMatrixToFile(strcat(strcpy(fullpath, outputdir), "ftype_sim.txt"), &ftype_simPtr);

    
    return 0;
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
