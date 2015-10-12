#include <stdio.h> 
#include <stdlib.h> 
#include <sys/time.h> 
#include <papi.h>


#define NUM_EVENTS 3
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }





/*@ global @*/
/*@ external @*/

extern double getClock(); 

int main(int argc, char *argv[]) {
  /*@ declarations @*/  
  /*@ prologue @*/



	///////////////////////////////////////////////////////////////////////
	//papi 
	
	int events[NUM_EVENTS] = {PAPI_LD_INS,PAPI_SR_INS,PAPI_FP_INS};
	long long  values[NUM_EVENTS];
	char errstring[PAPI_MAX_STR_LEN];
	int retval;


	if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
	{
		fprintf(stderr, "Error: %d %s\n",retval, errstring);
		exit(1);
	}



	//////////////////////////////////////////////////////////////////////

	int orio_i;
	int papi_j;
	double tot_time;

 	 /*
  	 Coordinate: /*@ coordinate @*/ 


	//Hacky way to cirumvent orio - but it works
	/*@ begin inner measurement @*/
	/*@ end inner measurement @*/
	/*@ end outer measurement @*/
	/*@ begin outer measurement @*/
  	*/

	
	for (orio_i=0; orio_i<ORIO_REPS; orio_i++) {
		tot_time = 0;
		for(papi_j=0; papi_j<NUM_EVENTS; papi_j++){


			orio_t_start = getClock();

			if ( (retval = PAPI_start_counters(events + papi_j,1)) != PAPI_OK)
				ERROR_RETURN(retval);

			
    			/*@ tested code @*/
			

			if ((retval=PAPI_stop_counters(values + papi_j,1)) != PAPI_OK)
				ERROR_RETURN(retval);


			orio_t_end = getClock();
 			orio_t = orio_t_end - orio_t_start;
 			tot_time +=orio_t;




			if (orio_i==0 && papi_j==0) {
     			 /*@ validation code @*/
			}
		}

		long long mem_ops = (values[0] + values[1])*sizeof(double);
		long long fp_ops = values[2] ;
		double arithmetic_int  = (double)fp_ops/(double)mem_ops;
		double gflops = (fp_ops*NUM_EVENTS)/(tot_time* 1e9);
          	printf("{'/*@ coordinate @*/' : [%f, %f] }\n", arithmetic_int,gflops);


		
	}
	


 	/*@ epilogue @*/
	return 0;
}