#include <stdio.h> 
#include <stdlib.h> 
#include <sys/time.h> 
#include <TAU.h>

#include <CL/opencl.hpp>

#define ORIO_OPENCL 1

/*@ global @*/ 
/*@ external @*/

//int main(int argc, char * argv[]) { 
  /*@ declarations @*/

  TAU_INIT(&argc, &argv)
  TAU_PROFILE_TIMER(orio_maintimer, "main()", "int (int, char **)", TAU_USER);
  TAU_PROFILE_START(orio_maintimer);
  TAU_PROFILE_SET_NODE(0);
  /*@ prologue @*/ 

  void * orio_profiler;
  TAU_PROFILER_CREATE(orio_profiler, "orio_generated_code", "", TAU_USER);
#ifdef ORIO_OPENCL
  void * compile_profiler;
  TAU_PROFILER_CREATE(compile_profiler, "orio_opencl_compile", "", TAU_USER);
  void * execute_profiler;
  TAU_PROFILER_CREATE(execute_profiler, "orio_opencl_execute", "", TAU_USER);
#endif

  int orio_i;
  for (orio_i=0; orio_i<ORIO_REPS; orio_i++) 
  { 
    TAU_PROFILER_START(orio_profiler);   
    /*@ tested code @*/
    TAU_PROFILER_STOP(orio_profiler);
    if(orio_i==0) {
      /*@ validation code @*/
    }
  } 
  double orio_inclusive[TAU_MAX_COUNTERS];
#ifdef ORIO_OPENCL
  TAU_PROFILER_GET_INCLUSIVE_VALUES(execute_profiler, &orio_inclusive);
#else
  TAU_PROFILER_GET_INCLUSIVE_VALUES(orio_profiler, &orio_inclusive);
#endif
  printf("{'/*@ coordinate @*/' : %g}\n", orio_inclusive[0]);
   
  /*@ epilogue @*/ 
  TAU_PROFILE_STOP(orio_maintimer);

  return 0; 
}
