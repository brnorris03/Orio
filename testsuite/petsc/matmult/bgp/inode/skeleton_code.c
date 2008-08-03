#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "mpi.h"

#include <omp.h>

/*@ global @*/  

#define max(x,y) ((x) > (y)? (x) : (y))
#define min(x,y) ((x) < (y)? (x) : (y))

#define SPRN_TBRL 0x10C  // Time Base Read Lower Register (user & sup R/O)
#define SPRN_TBRU 0x10D  // Time Base Read Upper Register (user & sup R/O)
#define _bgp_mfspr( SPRN )\
({\
  unsigned int tmp;\
  do {\
    asm volatile ("mfspr %0,%1" : "=&r" (tmp) : "i" (SPRN) : "memory" );\
  }\
  while(0);\
  tmp;\
})\

double getClock()
{
    union {
	unsigned int ul[2];
	unsigned long long ull;
    }
    hack;
    unsigned int utmp;
    do {
	utmp = _bgp_mfspr( SPRN_TBRU );
	hack.ul[1] = _bgp_mfspr( SPRN_TBRL );
	hack.ul[0] = _bgp_mfspr( SPRN_TBRU );
    }
    while(utmp != hack.ul[0]);
    return((double) hack.ull );
}

typedef struct {
    int testid;
    char coord[1024];
    double tm;
} TimingInfo;

int main(int argc, char *argv[])
{
    int numprocs, myid, _i;
    TimingInfo mytimeinfo;
    TimingInfo *timevec;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    /* Construct the MPI type for the timing info (what a pain!) */
    MPI_Datatype TimingInfoMPIType;
    {
	MPI_Datatype type[3] = {MPI_INT, MPI_CHAR, MPI_DOUBLE};
	int blocklen[3] = {1,1024,1};
	MPI_Aint disp[3];
	int base;
	MPI_Address( &mytimeinfo.testid, disp);
	MPI_Address( &mytimeinfo.coord, disp+1);
	MPI_Address( &mytimeinfo.tm, disp+2);
	base = disp[0];
	for (_i=0; _i <3; _i++) disp[_i] -= base;
	MPI_Type_struct( 3, blocklen, disp, type, &TimingInfoMPIType);
	MPI_Type_commit( &TimingInfoMPIType);
    }
    /* end of MPI type construction */

    if (myid == 0) timevec = (TimingInfo*) malloc(numprocs * sizeof(TimingInfo));

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

    switch (myid)
    {
	/*@ begin switch body @*/
	double orio_t_start, orio_t_end, orio_t_total=0;
	int orio_i;
	mytimeinfo.testid = myid;
	strcpy(mytimeinfo.coord,"/*@ coordinate @*/");
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
	    
	    orio_t_start = getClock();
	    
	    /*@ tested code @*/

	    orio_t_end = getClock();
	    orio_t_total += orio_t_end - orio_t_start;
	}
	orio_t_total = orio_t_total / REPS;
	mytimeinfo.tm = orio_t_total;
	/*@ end switch body @*/

	default:
	    mytimeinfo.testid = -1;
	    strcpy(mytimeinfo.coord,"");
	    mytimeinfo.tm = -1;
	    break;
    }

    MPI_Gather(&mytimeinfo, 1, TimingInfoMPIType, timevec, 1, TimingInfoMPIType, 0, MPI_COMM_WORLD);

    if (myid==0) {
	printf("{");
	if (mytimeinfo.tm >= 0 && strcmp(mytimeinfo.coord, "") != 0)
	    printf(" '%s' : %g,", mytimeinfo.coord, mytimeinfo.tm);
	for (_i=1; _i<numprocs; _i++) {
	    if (timevec[_i].tm >= 0 && strcmp(timevec[_i].coord, "") != 0)
		printf(" '%s' : %g,", timevec[_i].coord, timevec[_i].tm);
	}
	printf("}\n");
    }

    MPI_Finalize();

    /*@ epilogue @*/

    return 0;
}

