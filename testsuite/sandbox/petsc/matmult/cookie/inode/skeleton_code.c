#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#include <omp.h>
#include <papi.h>

#define max(x,y) ((x) > (y)? (x) : (y))
#define min(x,y) ((x) < (y)? (x) : (y))

/*@ global @*/  

int main()
{
    /*@ prologue @*/

    // Start of input declarations
    int gnrows = G_NROWS;
    int gncols = G_NCOLS;
    int bnrows = B_NROWS;
    int bncolsmin = B_NCOLS_MIN;
    int bncolsmax = B_NCOLS_MAX;
    int bncolsstride = B_NCOLS_STRIDE;
    if (gnrows % bnrows != 0) {
	printf("error: the global number of rows must be divisible by the block number of rows.\n");
	exit(1);
    }
    if ((bncolsmax-bncolsmin) % bncolsstride != 0) {
	printf("error: the length of the column range must be divisible by the column stride\n");
	exit(1);
    }
    double *y = (double *) malloc(gnrows*sizeof(double)); 
    double *x = (double *) malloc(gncols*sizeof(double)); 
    double *aa = (double *) malloc(gnrows*bncolsmax*sizeof(double)); 
    int *ai = (int *) malloc((gnrows+1)*sizeof(int)); 
    int *aj = (int *) malloc(gnrows*bncolsmax*sizeof(int)); 
    int total_rows = gnrows; 
    int total_inodes = total_rows/bnrows; 
    int *inode_sizes = (int *) malloc(total_inodes*sizeof(int)); 
    int *inode_rows = (int *) malloc((total_inodes+1)*sizeof(int)); 
    { 
	int i,j,k; 
	for (i=0; i<gnrows; i++) 
	    y[i]=0; 
	for (i=0; i<gncols; i++) 
	    x[i]=(i%10)+1; 
	for (i=0; i<total_inodes; i++) { 
	    inode_sizes[i]=bnrows; 
	    inode_rows[i]=i*bnrows; 
	} 
	inode_rows[total_inodes]=total_inodes*bnrows; 
	int ind=0; 
	for (i=0; i<gnrows; i+=bnrows) { 
	    int cur_bncols=bncolsmin+((i/bnrows)*bncolsstride)% 
		((((bncolsmax-bncolsmin)/bncolsstride)+1)*bncolsstride); 
	    for (j=0; j<bnrows; j++) { 
		ai[i+j]=ind; 
		for (k=0; k<cur_bncols; k++) { 
		    aa[ind]=(((j+1)*k)%10)+j+2; 
		    if (j==0) 
			aj[ind]=(i/bnrows)*((gncols-bncolsmax)/total_inodes)+k; 
		    else 
			aj[ind]=aj[ind-cur_bncols]; 
		    ind++; 
		} 
	    } 
	} 
	ai[gnrows]=ind; 
    }
    double *y_t = y;
    double *x_t = x;
    double *aa_t = aa;
    int *ai_t = ai;
    int *aj_t = aj;
    int total_rows_t = total_rows;
    int total_inodes_t = total_inodes;
    int *inode_sizes_t = inode_sizes;
    int *inode_rows_t = inode_rows;
    // End of input declarations

    long long orio_total_cycles = 0;
    long long orio_avg_cycles;
    int orio_i;

    for (orio_i=0; orio_i<REPS; orio_i++)
      {
	y = y_t;
	x = x_t;
	aa = aa_t;
	ai = ai_t;
	aj = aj_t;
	total_rows = total_rows_t;
	total_inodes = total_inodes_t;
	inode_sizes = inode_sizes_t;
	inode_rows = inode_rows_t;

	int err, EventSet = PAPI_NULL;
	long long CtrValues[1];
	err = PAPI_library_init(PAPI_VER_CURRENT);
	if (err != PAPI_VER_CURRENT) {
	  printf("PAPI library initialization error!\n");
	  exit(1);
	}
	if (PAPI_create_eventset(&EventSet) != PAPI_OK) {
	  printf("Failed to create PAPI Event Set\n");
	  exit(1);
	}
	if (PAPI_query_event(PAPI_TOT_CYC) == PAPI_OK)
	  err = PAPI_add_event(EventSet, PAPI_TOT_CYC);
	if (err != PAPI_OK) {
	  printf("Failed to add PAPI event\n");
	  exit(1);
	}
	PAPI_start(EventSet);

	/*@ tested code @*/

	PAPI_stop(EventSet, &CtrValues[0]);
	orio_total_cycles += CtrValues[0];
      }

    orio_avg_cycles = orio_total_cycles / REPS;

    printf("{'/*@ coordinate @*/' : %ld}", orio_avg_cycles); 

    /*@ epilogue @*/

    return y[0]; // to avoid the dead code elimination                                                   
}


