#include <stdio.h>  
#include <stdlib.h> 
#include <sys/time.h>   

#include <papi.h>

#define TROWS 100000
#define TCOLS 100000
#define BROWS 4
#define BCOLS 20
#include "decl_code.h"

#include "init_code.c"

  

#define max(x,y) ((x) > (y)? (x) : (y))
#define min(x,y) ((x) < (y)? (x) : (y))

int main()  
{
  init_input_vars();
  
  
  int trows = TROWS;
  int tcols = TCOLS;
  int brows = BROWS;
  int bcols = BCOLS;
  
  if (trows % brows != 0) {
    printf("error: the global number of rows must be divisible by the block number of rows.\n");
    exit(1);
  }

  double *y = (double *) malloc(trows*sizeof(double));
  double *x = (double *) malloc(tcols*sizeof(double));
  double *aa = (double *) malloc(trows*(bcols+100)*sizeof(double));
  int *ai = (int *) malloc(trows*sizeof(int));
  int *aj = (int *) malloc(trows*(bcols+100)*sizeof(int));
  int *node_sizes = (int *) malloc(trows*sizeof(int));
  int node_max = 0;
  {
    int i,j,k,ind;
    for (i=0; i<trows; i++)
      y[i]=0;
    for (i=0; i<tcols; i++)
      x[i]=(i%10)+1;
    ind=0;
    for (i=0; i<trows; i+=brows) {
      node_max++;
      node_sizes[i/brows]=brows;
      for (j=0; j<brows; j++) {
	ai[i+j]=ind;
	int start_pos=max((((int)(1.0*i/trows*tcols))-5*bcols),0);
	int mult=((5*(j+3))%3)+1;
	for (k=0; k<bcols; k++) {
	  aa[ind]=(((j+1)*k)%10)+j+2;
	  if (j==0)
	    aj[ind]=start_pos+k*mult;
	  else
	    aj[ind]=aj[ind-bcols];
	  ind++;
	}
      }
    }
  }
  double *y_o = y;
  double *x_o = x;
  double *aa_o = aa;
  int *ai_o = ai;
  int *aj_o = aj;
  int *node_sizes_o = node_sizes;

  long long orio_total_cycles = 0;
  long long orio_avg_cycles;
  int orio_i;

  for (orio_i=0; orio_i<REPS; orio_i++) 
    {  
      y=y_o; x=x_o; aa=aa_o; ai=ai_o; aj=aj_o; node_sizes=node_sizes_o;
      
      int *ns=node_sizes;
      double *v1,*v2,*v3,*v4,*v5;
      int i,row,n,*ii,*idx,nsz,sz,i1,i2;
      double sum1,sum2,sum3,sum4,sum5,tmp0,tmp1;
      
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
      
      v1=aa;
      ii=ai;
      idx=aj;
      for (i=0,row=0; i<node_max; ++i){
	nsz  = ns[i];
	n    = ii[1] - ii[0];
	ii  += nsz;
	sz   = n;
	switch (nsz){               /* Each loop in 'case' is unrolled */
	case 1 :
	  sum1  = 0;
	  for(n = 0; n< sz-1; n+=2) {
	    i1   = idx[0];          /* The instructions are ordered to */
	    i2   = idx[1];          /* make the compiler's job easy */
	    idx += 2;
	    tmp0 = x[i1];
	    tmp1 = x[i2];
	    sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
	  }
	  if (n == sz-1){          /* Take care of the last nonzero  */
	    tmp0  = x[*idx++];
	    sum1 += *v1++ * tmp0;
	  }
	  y[row++]=sum1;
	  break;
	case 2:
	  sum1  = 0;
	  sum2  = 0;
	  v2    = v1 + n;
	  for (n = 0; n< sz-1; n+=2) {
	    i1   = idx[0];
	    i2   = idx[1];
	    idx += 2;
	    tmp0 = x[i1];
	    tmp1 = x[i2];
	    sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
	    sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
	  }
	  if (n == sz-1){
	    tmp0  = x[*idx++];
	    sum1 += *v1++ * tmp0;
	    sum2 += *v2++ * tmp0;
	  }
	  y[row++]=sum1;
	  y[row++]=sum2;
	  v1      =v2;              /* Since the next block to be processed starts there*/
	  idx    +=sz;
	  break;
	case 3:
	  sum1  = 0;
	  sum2  = 0;
	  sum3  = 0;
	  v2    = v1 + n;
	  v3    = v2 + n;
	  for (n = 0; n< sz-1; n+=2) {
	    i1   = idx[0];
	    i2   = idx[1];
	    idx += 2;
	    tmp0 = x[i1];
	    tmp1 = x[i2];
	    sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
	    sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
	    sum3 += v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
	  }
	  if (n == sz-1){
	    tmp0  = x[*idx++];
	    sum1 += *v1++ * tmp0;
	    sum2 += *v2++ * tmp0;
	    sum3 += *v3++ * tmp0;
	  }
	  y[row++]=sum1;
	  y[row++]=sum2;
	  y[row++]=sum3;
	  v1       =v3;             /* Since the next block to be processed starts there*/
	  idx     +=2*sz;
	  break;
	case 4:
	  sum1  = 0;
	  sum2  = 0;
	  sum3  = 0;
	  sum4  = 0;
	  v2    = v1 + n;
	  v3    = v2 + n;
	  v4    = v3 + n;
	  for (n = 0; n< sz-1; n+=2) {
	    i1   = idx[0];
	    i2   = idx[1];
	    idx += 2;
	    tmp0 = x[i1];
	    tmp1 = x[i2];
	    sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
	    sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
	    sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
	    sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
	  }
	  if (n == sz-1){
	    tmp0  = x[*idx++];
	    sum1 += *v1++ * tmp0;
	    sum2 += *v2++ * tmp0;
	    sum3 += *v3++ * tmp0;
	    sum4 += *v4++ * tmp0;
	  }
	  y[row++]=sum1;
	  y[row++]=sum2;
	  y[row++]=sum3;
	  y[row++]=sum4;
	  v1      =v4;              /* Since the next block to be processed starts there*/
	  idx    +=3*sz;
	  break;
	case 5:
	  sum1  = 0;
	  sum2  = 0;
	  sum3  = 0;
	  sum4  = 0;
	  sum5  = 0;
	  v2    = v1 + n;
	  v3    = v2 + n;
	  v4    = v3 + n;
	  v5    = v4 + n;
	  for (n = 0; n<sz-1; n+=2) {
	    i1   = idx[0];
	    i2   = idx[1];
	    idx += 2;
	    tmp0 = x[i1];
	    tmp1 = x[i2];
	    sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
	    sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
	    sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
	    sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
	    sum5 += v5[0] * tmp0 + v5[1] *tmp1; v5 += 2;
	  }
	  if (n == sz-1){
	    tmp0  = x[*idx++];
	    sum1 += *v1++ * tmp0;
	    sum2 += *v2++ * tmp0;
	    sum3 += *v3++ * tmp0;
	    sum4 += *v4++ * tmp0;
	    sum5 += *v5++ * tmp0;
	  }
	  y[row++]=sum1;
	  y[row++]=sum2;
	  y[row++]=sum3;
	  y[row++]=sum4;
	  y[row++]=sum5;
	  v1      =v5;       /* Since the next block to be processed starts there */
	  idx    +=4*sz;
	  break;
	default :
	  printf("error: node size not yet supported\n");
	  exit(1);
	}
      }


      PAPI_stop(EventSet, &CtrValues[0]);
      orio_total_cycles += CtrValues[0];
    }  
  
  orio_avg_cycles = orio_total_cycles / REPS;
  printf("%ld\n", orio_avg_cycles);

  
  return y[0]; // to avoid the dead code elimination
}   

