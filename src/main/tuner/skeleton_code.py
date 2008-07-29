#
# The skeleton code used for performance testing
#

import os, re, sys

#-----------------------------------------------------

SEQ_DEFAULT = r'''

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/*@ global @*/

#ifdef BGP_COUNTER
#define SPRN_TBRL 0x10C // Time Base Read Lower Register (user & sup R/O)
#define SPRN_TBRU 0x10D // Time Base Read Upper Register (user & sup R/O)
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
#else
double getClock()
{
  struct timezone tzp;
  struct timeval tp;
  int stat;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}
#endif

int main()
{
  /*@ prologue @*/

  double orio_t_start, orio_t_end, orio_t_total=0;
  int orio_i;

  for (orio_i=0; orio_i<REPS; orio_i++)
  {
    orio_t_start = getClock();
    
    /*@ tested code @*/

    orio_t_end = getClock();
    orio_t_total += orio_t_end - orio_t_start;
  }
  orio_t_total = orio_t_total / REPS;
  
  printf("{'' : %g}", orio_t_total);
  
  /*@ epilogue @*/

  return 0;
}

'''

#-----------------------------------------------------

PAR_DEFAULT = r'''

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "mpi.h"

/*@ global @*/

#ifdef BGP_COUNTER
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
#else
double getClock()
{
  struct timezone tzp;
  struct timeval tp;
  int stat;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}
#endif

typedef struct {
  int testid;
  char coord[1024];
  double tm;
} TimingInfo;

int main()
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
  
  switch (myid)
  {
      /*@ begin switch body @*/
      double orio_t_start, orio_t_end, orio_t_total=0;
      int orio_i;
      mytimeinfo.testid = myid;
      strcpy(mytimeinfo.coord,"/*@ coordinate @*/");
      for (orio_i=0; orio_i<REPS; orio_i++)
      {
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
    printf("{'%s' : %g", mytimeinfo.coord, mytimeinfo.tm);
    for (_i=1; _i<numprocs; _i++) {
      printf(", '%s' : %g", timevec[_i].coord, timevec[_i].tm);
    }
    printf("}\n");
  }

  MPI_Finalize();

  /*@ epilogue @*/

  return 0;
}

'''

#-----------------------------------------------------

class PerfTestSkeletonCode:
    '''The skeleton code used in the performance testing'''

    # tags
    __GLOBAL_TAG = r'/\*@\s*global\s*@\*/'
    __PROLOGUE_TAG = r'/\*@\s*prologue\s*@\*/'
    __EPILOGUE_TAG = r'/\*@\s*epilogue\s*@\*/'
    __TCODE_TAG = r'/\*@\s*tested\s+code\s*@\*/'
    __COORD_TAG = r'/\*@\s*coordinate\s*@\*/'
    __BEGIN_SWITCHBODY_TAG = r'/\*@\s*begin\s+switch\s+body\s*@\*/'
    __END_SWITCHBODY_TAG = r'/\*@\s*end\s+switch\s+body\s*@\*/'
    __SWITCHBODY_TAG = __BEGIN_SWITCHBODY_TAG + r'((.|\n)*?)' + __END_SWITCHBODY_TAG

    #-----------------------------------------------------
    
    def __init__(self, code, use_parallel_search):
        '''To instantiate the skeleton code for the performance testing'''

        if code == None:
            if use_parallel_search:
                code = PAR_DEFAULT
            else:
                code = SEQ_DEFAULT

        self.code = code
        self.use_parallel_search = use_parallel_search

        self.__checkSkeletonCode(self.code)

    #-----------------------------------------------------

    def __checkSkeletonCode(self, code):
        '''To check the validity of the skeleton code'''

        match_obj = re.search(self.__GLOBAL_TAG, code)
        if not match_obj:
            print 'error: missing "global" tag in the skeleton code'
            sys.exit(1)

        match_obj = re.search(self.__PROLOGUE_TAG, code)
        if not match_obj:
            print 'error: missing "prologue" tag in the skeleton code'
            sys.exit(1)

        match_obj = re.search(self.__EPILOGUE_TAG, code)
        if not match_obj:
            print 'error: missing "epilogue" tag in the skeleton code'
            sys.exit(1)

        match_obj = re.search(self.__TCODE_TAG, code)
        if not match_obj:
            print 'error: missing "tested code" tag in the skeleton code'
            sys.exit(1)

        if self.use_parallel_search:

            match_obj = re.search(self.__COORD_TAG, code)
            if not match_obj:
                print 'error: missing "coordinate" tag in the skeleton code'
                sys.exit(1)
            
            match_obj = re.search(self.__BEGIN_SWITCHBODY_TAG, code)
            if not match_obj:
                print 'error: missing "begin switch body" tag in the skeleton code'
                sys.exit(1)
        
            match_obj = re.search(self.__END_SWITCHBODY_TAG, code)
            if not match_obj:
                print 'error: missing "end switch body" tag in the skeleton code'
                sys.exit(1)
        
            match_obj = re.search(self.__SWITCHBODY_TAG, code)
            if not match_obj:
                print 'internal error: missing placement of switch body statement'
                sys.exit(1)

            switch_body_code = match_obj.group(1)

            match_obj = re.search(self.__TCODE_TAG, switch_body_code)
            if not match_obj:
                print 'error: missing "tested code" tag in the switch body statement'
                sys.exit(1)
            
            match_obj = re.search(self.__COORD_TAG, switch_body_code)
            if not match_obj:
                print 'error: missing "coordinate" tag in the switch body statement'
                sys.exit(1)

    #-----------------------------------------------------

    def insertCode(self, global_code, prologue_code, epilogue_code, tested_code_map):
        '''To insert code fragments into the skeleton code'''

        # check the given tested code mapping
        if len(tested_code_map) == 0:
            print 'internal error: the number of tested codes cannot be zero'
            sys.exit(1)
        if not self.use_parallel_search and len(tested_code_map) != 1:
            print 'internal error: the number of tested sequential codes must be exactly one'
            sys.exit(1)

        # initialize the performance-testing code
        code = self.code

        # insert global definitions, prologue, and epilogue codes
        code = re.sub(self.__GLOBAL_TAG, global_code, code)
        code = re.sub(self.__PROLOGUE_TAG, prologue_code, code)
        code = re.sub(self.__EPILOGUE_TAG, epilogue_code, code)

        # insert the parallel code
        if self.use_parallel_search:
            switch_body_code = re.search(self.__SWITCHBODY_TAG, code).group(1)
            tcode = ''
            for i, (code_key, code_value) in enumerate(tested_code_map.items()):
                scode = switch_body_code
                scode = re.sub(self.__COORD_TAG, code_key, scode)
                scode = re.sub(self.__TCODE_TAG, code_value, scode)
                tcode += '\n'
                tcode += '  case %s:\n' % i
                tcode += '    {\n' + scode + '\n    }\n'
                tcode += '    break;\n'
            code = re.sub(self.__SWITCHBODY_TAG, tcode, code)
            
        # insert the sequential code
        else:
            (tcode,) = tested_code_map.values()
            code = re.sub(self.__TCODE_TAG, tcode, code)

        # return the performance-testing code
        return code

