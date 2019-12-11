#
# The skeleton code used for performance testing
#

import re, sys
from orio.main.util.globals import *

#-----------------------------------------------------
SEQ_TIMER = '''
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

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

double getClock() {
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
#if !defined(__APPLE__) && !defined(_OPENMP)
double getClock() {
    struct timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return -1;
    return (double)ts.tv_sec + ((double)ts.tv_nsec)*1.0e-9;
}
#else
double getClock() {
  struct timezone tzp;
  struct timeval tp;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}
#endif
#endif
'''

SEQ_DEFAULT = r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

/*@ global @*/
/*@ external @*/

extern double getClock(); 

//int main(int argc, char *argv[]) { // part of declaration generation
  /*@ declarations @*/  
  /*@ prologue @*/

  int orio_i;

  /*
   Coordinate: /*@ coordinate @*/ 
  */
  
  /*@ begin outer measurement @*/
  for (orio_i=0; orio_i<ORIO_REPS; orio_i++) {
    /*@ begin inner measurement @*/
    
    /*@ tested code @*/

    /*@ end inner measurement @*/
    if (orio_i==0) {
      /*@ validation code @*/
    }
  }
  /*@ end outer measurement @*/
  
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
#include <math.h>
#include "mpi.h"


/*@ global @*/
/*@ external @*/

#define BIG_NUMBER 147483647.0

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
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}
#endif

typedef struct {
  int testid;
  char coord[1024];
  double tm;
} TimingInfo;

//int main(int argc, char *argv[]) { // part of declaration generation
  /*@ declarations @*/
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
    MPI_Aint disp[3], base;
    MPI_Get_address( &mytimeinfo.testid, &disp[0]);
    MPI_Get_address( &mytimeinfo.coord, &disp[1]);
    MPI_Get_address( &mytimeinfo.tm, &disp[2]);
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
      double orio_t_start, orio_t_end, orio_t, orio_t_total=0, orio_t_min=BIG_NUMBER;
      int orio_i;
      mytimeinfo.testid = myid;
      strcpy(mytimeinfo.coord,"/*@ coordinate @*/");
      for (orio_i=0; orio_i<ORIO_REPS; orio_i++)
      {
        orio_t_start = getClock(); 

        /*@ tested code @*/

        orio_t_end = getClock();
        orio_t = orio_t_end - orio_t_start;
        if (orio_t < orio_t_min) orio_t_min = orio_t;
      }
      /* Mean of all times -- not a good idea in the presence of noise, instead use min */
      /* orio_t_total = orio_t_total / REPS; */
      orio_t_total = orio_t_min;
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

'''

SEQ_FORTRAN_DEFAULT = r'''

program main

    implicit none
    
    integer, parameter :: double = selected_real_kind(10,40)
    integer, parameter :: single = selected_real_kind(5,20)
    
    real(double) :: orio_t_start, orio_t_end, orio_min_time, orio_delta_time
    integer      :: orio_i
    
!@ declarations @!
!@ prologue @!
    
    orio_min_time = X'7FF00000'   ! large number
    do orio_i = 1, ORIO_REPS
    
      orio_t_start = getClock()
    
      !@ tested code @!
    
      orio_t_end = getClock()
      orio_delta_time = orio_t_end - orio_t_start
      if (orio_delta_time < orio_min_time) then
          orio_min_time = orio_delta_time
      end if
    
    enddo
    
    write(*,"(A,ES20.13,A)",advance="no") "{'!@ coordinate @!' : ", orio_delta_time, "}"
    
    !@ epilogue @!
    
    contains

    real(double) function getClock()
        implicit none
        integer (kind = 8) clock_count, clock_max, clock_rate
        integer ( kind = 8 ), parameter :: call_num = 100

        call system_clock(clock_count, clock_rate, clock_max)

        getClock = dble(clock_count) / dble(call_num * clock_rate)
    end function

end program main

'''

#-----------------------------------------------------

PAR_FORTRAN_DEFAULT = r'''

program main

    use mpi
    implicit none
    
    integer, parameter :: double = selected_real_kind(10,40)
    integer, parameter :: single = selected_real_kind(5,20)
    
    type TimingInfo
      sequence
      integer             :: testid
      character(len=1024) :: coord
      real(double)        :: tm
    end type TimingInfo
    
    integer          :: numprocs, myid, i_, ierror
    integer          :: TimingInfoMPIType
    integer          :: blocklen(3) = (/ 1, 1024, 1/)
    integer          :: disp(3)     = (/ 0, 4, 4+1024 /) ! assume four-byte integers
    integer          :: types(3)    = (/ MPI_INTEGER, MPI_CHARACTER, &
                                         MPI_DOUBLE_PRECISION /)
    type(TimingInfo) :: mytimeinfo
    type(TimingInfo), allocatable :: timevec(:)
    real(double)     :: orio_t_start, orio_t_end, orio_min_time, orio_delta_time
    integer          :: orio_i
    
!@ declarations @!
    
    call mpi_init(ierror)
    call mpi_comm_size(MPI_COMM_WORLD, numprocs)
    call mpi_comm_rank(MPI_COMM_WORLD, myid)
    
    ! Construct the MPI type for the timing info (what a pain!)
    
    call mpi_type_create_struct(3, blocklen, disp, types, TimingInfoMPIType, ierror)
    call mpi_type_commit(TimingInfoMPIType, ierror)
    
    if (myid == 0) allocate(timevec(0:numprocs-1))
    
    orio_min_time = X'7FF00000'   ! large number
    
!@ prologue @!
      
    select case (myid)
    
        !@ begin switch body @!
    
        mytimeinfo%testid = myid
        mytimeinfo%coord  = "!@ coordinate @!"
    
        do orio_i = 1, ORIO_REPS
    
          orio_t_start = MPI_Wtime()
    
          !@ tested code @!
    
          orio_t_end = MPI_Wtime()
          orio_min_time = min(orio_min_time, orio_t_end - orio_t_start)
        enddo
    
        mytimeinfo%tm = orio_min_time
    
        !@ end switch body @!
    
      case default
    
        mytimeinfo%testid = -1
        mytimeinfo%coord  = ""
        mytimeinfo%tm     = -1
    
    end select
    
    call mpi_gather(mytimeinfo, 1, TimingInfoMPIType, &
                    timevec, 1, TimingInfoMPIType, &
                    0, MPI_COMM_WORLD, ierror)
    
    if (myid == 0) then
      write(*,"(A)",advance="no") "{"
      if ((mytimeinfo%tm >= 0) .and. (mytimeinfo%coord /= "")) &
        write(*,"(3A,ES20.13)",advance="no") " '", mytimeinfo%coord, "' : ", &
                                             mytimeinfo%tm
      do i_ = 1, numprocs-1
        if ((timevec(i_)%tm >= 0) .and. (timevec(i_)%coord /= ""))
          write (*,"(3A,ES20.13)",advance="no") &
            " '", timevec(i_)%coord, "' : ", timevec(i_)%tm
      enddo
      write(*,"(A)",advance="yes") "}"
    endif
    
    call mpi_finalize(ierror)
    
    !@ epilogue @!

end program main
'''

#----------------------------------------------------------------------------------------------------------------------
SEQ_DEFAULT_CUDA = r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

/*@ global @*/
/*@ external @*/

int main(int argc, char *argv[]) {
  /*@ declarations @*/
  /*@ prologue @*/
  cudaSetDeviceFlags(cudaDeviceBlockingSync);
  float orcu_elapsed=0.0, orcu_transfer=0.0;
  cudaEvent_t tstart, tstop, start, stop;
  cudaEventCreate(&tstart); cudaEventCreate(&tstop);
  cudaEventCreate(&start);  cudaEventCreate(&stop);
  /*@ begin outer measurement @*/
  for (int orio_i=0; orio_i<ORIO_REPS; orio_i++) {
    /*@ begin inner measurement @*/
    
    /*@ tested code @*/

    /*@ end inner measurement @*/
    printf("{'/*@ coordinate @*/' : (%g,%g)}\n", orcu_elapsed, orcu_transfer);
  }
  /*@ end outer measurement @*/
  cudaEventDestroy(tstart); cudaEventDestroy(tstop);
  cudaEventDestroy(start);  cudaEventDestroy(stop);
  
  /*@ epilogue @*/
  return 0;
}
'''
#----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------

class PerfTestSkeletonCode:
    '''The skeleton code used in the performance testing'''

    # tags
    __GLOBAL_TAG = r'/\*@\s*global\s*@\*/'
    __EXTERNAL_TAG = r'/\*@\s*external\s*@\*/'
    __DECLARATIONS_TAG = r'/\*@\s*declarations\s*@\*/'
    __PROLOGUE_TAG = r'/\*@\s*prologue\s*@\*/'
    __EPILOGUE_TAG = r'/\*@\s*epilogue\s*@\*/'
    __TCODE_TAG = r'/\*@\s*tested\s+code\s*@\*/'
    __BEGIN_INNER_MEASURE_TAG = r'/\*@\s*begin\s+inner\s+measurement\s*@\*/'
    __END_INNER_MEASURE_TAG = r'/\*@\s*end\s+inner\s+measurement\s*@\*/'
    __BEGIN_OUTER_MEASURE_TAG = r'/\*@\s*begin\s+outer\s+measurement\s*@\*/'
    __END_OUTER_MEASURE_TAG = r'/\*@\s*end\s+outer\s+measurement\s*@\*/'
    __VALIDATION_TAG = r'/\*@\s*validation\s+code\s*@\*/'
    __COORD_TAG = r'/\*@\s*coordinate\s*@\*/'
    __BEGIN_SWITCHBODY_TAG = r'/\*@\s*begin\s+switch\s+body\s*@\*/'
    __END_SWITCHBODY_TAG = r'/\*@\s*end\s+switch\s+body\s*@\*/'
    __SWITCHBODY_TAG = __BEGIN_SWITCHBODY_TAG + r'((.|\n)*?)' + __END_SWITCHBODY_TAG

    #-----------------------------------------------------
    
    def __init__(self, code, use_parallel_search, language='c'):
        '''To instantiate the skeleton code for the performance testing'''

        if code == None:
            if use_parallel_search:
                code = PAR_DEFAULT
            else:
                if language == 'c':
                    code = SEQ_DEFAULT
                else:
                    code = SEQ_DEFAULT_CUDA

        self.code = code
        self.use_parallel_search = use_parallel_search
        self.language = language

        self.__checkSkeletonCode(self.code)

    #-----------------------------------------------------

    def __checkSkeletonCode(self, code):
        '''To check the validity of the skeleton code'''

        match_obj = re.search(self.__GLOBAL_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code:  missing "global" tag in the skeleton code')

        match_obj = re.search(self.__EXTERNAL_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code:  missing "external" tag in the skeleton code')

        match_obj = re.search(self.__DECLARATIONS_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code:  missing "declarations" tag in the skeleton code')

        match_obj = re.search(self.__PROLOGUE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code:  missing "prologue" tag in the skeleton code')

        match_obj = re.search(self.__EPILOGUE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code:  missing "epilogue" tag in the skeleton code')

        match_obj = re.search(self.__TCODE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code:  missing "tested code" tag in the skeleton code')
            
        match_obj = re.search(self.__BEGIN_INNER_MEASURE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code: missing "begin inner measurement" tag in the skeleton code')

        match_obj = re.search(self.__END_INNER_MEASURE_TAG,code)
        if not match_obj:
            err('main.tuner.skeleton_code: missing "end inner measurement" tag in the skeleton code')

        match_obj = re.search(self.__BEGIN_OUTER_MEASURE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code: missing "begin outer measurement" tag in the skeleton code')

        match_obj = re.search(self.__END_OUTER_MEASURE_TAG,code)
        if not match_obj:
            err('main.tuner.skeleton_code: missing "end outer measurement" tag in the skeleton code')


        match_obj = re.search(self.__VALIDATION_TAG, code)
        if not match_obj:
            warn('main.tuner.skeleton_code:  missing "validation code" tag in the skeleton code')

        match_obj = re.search(self.__COORD_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code:  missing "coordinate" tag in the skeleton code')
            
        if self.use_parallel_search:

            match_obj = re.search(self.__BEGIN_SWITCHBODY_TAG, code)
            if not match_obj:
                err('main.tuner.skeleton_code:  missing "begin switch body" tag in the skeleton code')
        
            match_obj = re.search(self.__END_SWITCHBODY_TAG, code)
            if not match_obj:
                err('main.tuner.skeleton_code:  missing "end switch body" tag in the skeleton code')
        
            match_obj = re.search(self.__SWITCHBODY_TAG, code)
            if not match_obj:
                err('main.tuner.skeleton_code internal error:  missing placement of switch body statement')

            switch_body_code = match_obj.group(1)

            match_obj = re.search(self.__TCODE_TAG, switch_body_code)
            if not match_obj:
                err('main.tuner.skeleton_code:  missing "tested code" tag in the switch body statement')
            
            match_obj = re.search(self.__VALIDATION_TAG, switch_body_code)
            if not match_obj:
                warn('main.tuner.skeleton_code:  missing "validation code" tag in the switch body statement')
            
            match_obj = re.search(self.__COORD_TAG, switch_body_code)
            if not match_obj:
                err('main.tuner.skeleton_code:  missing "coordinate" tag in the switch body statement')

    #-----------------------------------------------------

    def insertCode(self, global_code, prologue_code, epilogue_code, validation_code, 
                   begin_inner_measure_code, end_inner_measure_code, 
                   begin_outer_measure_code, end_outer_measure_code, 
                   tested_code_map):
        '''
        Insert code fragments into the skeleton driver code.
        
        @return: Complete specialized C source code string for the performance testing driver.
        
        @param global_code:
        @param prologue_code: 
        @param epilogue code:
        @param validation_code: 
        @param begin_inner_measure_code: start inner loop measurement, e.g., initialze time variable
        @param end_inner_measure_code: stop inner loop measurement, e.g., get time and find elapsed time value
        @param begin_outer_measure_code: start measurement around repetitions loop, e.g., initialze time variable
        @param end_outer_measure_code: stop measurement around repetitions loop, e.g., get time and find elapsed time value
        @param tested_code_map:
        '''

        # check the given tested code mapping
        if len(tested_code_map) == 0:
            err('main.tuner.skeleton_code internal error:  the number of tested codes cannot be zero')
        if not self.use_parallel_search and len(tested_code_map) != 1:
            err('main.tuner.skeleton_code internal error:  the number of tested sequential codes must be exactly one')

        # initialize the performance-testing code
        code = self.code

        # add cuda kernel definitions if any
        g = Globals()
        if self.language == 'cuda' and len(g.cunit_declarations) > 0:
            global_code += reduce(lambda x,y: x + y, g.cunit_declarations)
            g.cunit_declarations = []
            
        # TODO: make this less ugly
        # Declarations that must be in main() scope (not global)
        declarations_code = '\n#ifdef MAIN_DECLARATIONS\n  MAIN_DECLARATIONS()\n#endif'
        
        # insert global definitions, prologue, and epilogue codes
        code = re.sub(self.__GLOBAL_TAG, global_code, code)
        code = re.sub(self.__DECLARATIONS_TAG, declarations_code, code)
        code = re.sub(self.__PROLOGUE_TAG, prologue_code, code)
        code = re.sub(self.__EPILOGUE_TAG, epilogue_code, code)

        # insert the parallel code
        if self.use_parallel_search:
            switch_body_code = re.search(self.__SWITCHBODY_TAG, code).group(1)
            tcode = ''
            par_externals = ''
            for i, (code_key, (code_value, externals)) in enumerate(tested_code_map.items()):
                scode = switch_body_code
                scode = re.sub(self.__COORD_TAG, code_key, scode)
                scode = re.sub(self.__TCODE_TAG, code_value, scode)
                tcode += '\n'
                tcode += '  case %s:\n' % i
                tcode += '    {\n' + scode + '\n    }\n'
                tcode += '    break;\n'
                par_externals += externals
            code = re.sub(self.__EXTERNAL_TAG, par_externals, code)
            code = re.sub(self.__SWITCHBODY_TAG, tcode, code)
            
        # insert the sequential code
        else:
            ((coord_key, (tcode, externals)),) = tested_code_map.items()
            # TODO: customizable timing code for parallel cases
            code = re.sub(self.__BEGIN_INNER_MEASURE_TAG, begin_inner_measure_code, code)
            code = re.sub(self.__END_INNER_MEASURE_TAG, re.sub(self.__COORD_TAG, coord_key, end_inner_measure_code), code)
            code = re.sub(self.__BEGIN_OUTER_MEASURE_TAG, begin_outer_measure_code, code)
            code = re.sub(self.__END_OUTER_MEASURE_TAG, re.sub(self.__COORD_TAG, coord_key, end_outer_measure_code), code)
            code = re.sub(self.__EXTERNAL_TAG, externals, code)
            code = re.sub(self.__COORD_TAG, coord_key, code)
            code = re.sub(self.__TCODE_TAG, tcode, code)

        # insert the validation code
        code = re.sub(self.__VALIDATION_TAG, validation_code, code)

        # return the performance-testing code
        return code


class PerfTestSkeletonCodeFortran:
    '''The skeleton code used in the performance testing'''

    # tags
    __PROLOGUE_TAG = r'!@\s*prologue\s*@!'
    __DECLARATIONS_TAG = r'!@\s*declarations\s*@!'
    __ALLOCATIONS_TAG = r'!@\s*allocation\s*@!'
    __EPILOGUE_TAG = r'!@\s*epilogue\s*@!'
    __TCODE_TAG = r'!@\s*tested\s+code\s*@!'
    __COORD_TAG = r'!@\s*coordinate\s*@!'
    __BEGIN_SWITCHBODY_TAG = r'!@\s*begin\s+switch\s+body\s*@!'
    __END_SWITCHBODY_TAG = r'!@\s*end\s+switch\s+body\s*@!'
    __SWITCHBODY_TAG = __BEGIN_SWITCHBODY_TAG + r'((.|\n)*?)' + __END_SWITCHBODY_TAG

    #-----------------------------------------------------
    
    def __init__(self, code, use_parallel_search):
        '''To instantiate the skeleton code for the performance testing'''

        if code == None:
            if use_parallel_search:
                code = PAR_FORTRAN_DEFAULT
            else:
                code = SEQ_FORTRAN_DEFAULT

        self.code = code
        self.use_parallel_search = use_parallel_search

        self.__checkSkeletonCode(self.code)

    #-----------------------------------------------------

    def __checkSkeletonCode(self, code):
        '''To check the validity of the skeleton code'''

        match_obj = re.search(self.__PROLOGUE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code: missing "prologue" tag in the skeleton code', doexit=True)

        match_obj = re.search(self.__EPILOGUE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code: missing "epilogue" tag in the skeleton code')

        match_obj = re.search(self.__TCODE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code: missing "tested code" tag in the skeleton code')

        match_obj = re.search(self.__COORD_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code: missing "coordinate" tag in the skeleton code')
            
        if self.use_parallel_search:

            match_obj = re.search(self.__BEGIN_SWITCHBODY_TAG, code)
            if not match_obj:
                err('main.tuner.skeleton_code: : missing "begin switch body" tag in the skeleton code')
        
            match_obj = re.search(self.__END_SWITCHBODY_TAG, code)
            if not match_obj:
                err('main.tuner.skeleton_code: missing "end switch body" tag in the skeleton code')
        
            match_obj = re.search(self.__SWITCHBODY_TAG, code)
            if not match_obj:
                err('main.tuner.skeleton_code: internal error: missing placement of switch body statement')

            switch_body_code = match_obj.group(1)

            match_obj = re.search(self.__TCODE_TAG, switch_body_code)
            if not match_obj:
                err('main.tuner.skeleton_code: missing "tested code" tag in the switch body statement')
            
            match_obj = re.search(self.__COORD_TAG, switch_body_code)
            if not match_obj:
                err('main.tuner.skeleton_code: missing "coordinate" tag in the switch body statement')

    #-----------------------------------------------------

    def insertCode(self, decl_code, prologue_code, epilogue_code,
                   begin_inner_measure_code, end_inner_measure_code, 
                   begin_outer_measure_code, end_outer_measure_code, 
                   tested_code_map):
        '''To insert code fragments into the skeleton code'''

        # check the given tested code mapping
        if len(tested_code_map) == 0:
            err('main.tuner.skeleton_code: internal error: the number of tested codes cannot be zero')

        if not self.use_parallel_search and len(tested_code_map) != 1:
            err('main.tuner.skeleton_code: internal error: the number of tested sequential codes must be exactly one')

        # initialize the performance-testing code
        code = self.code

        # insert global definitions, prologue, and epilogue codes
        code = re.sub(self.__DECLARATIONS_TAG, decl_code, code)
        code = re.sub(self.__EPILOGUE_TAG, epilogue_code, code)
        
        # TODO: Insert profiling (e.g., timing) code
        code = re.sub(self.__BEGIN_INNER_MEASURE_TAG, begin_inner_measure_code, code)
        code = re.sub(self.__END_INNER_MEASURE_TAG, re.sub(self.__COORD_TAG, coord_key, end_inner_measure_code), code)
        code = re.sub(self.__BEGIN_OUTER_MEASURE_TAG, begin_outer_measure_code, code)
        code = re.sub(self.__END_OUTER_MEASURE_TAG, re.sub(self.__COORD_TAG, coord_key, end_outer_measure_code), code)

        # insert the parallel code
        if self.use_parallel_search:
            switch_body_code = re.search(self.__SWITCHBODY_TAG, code).group(1)
            tcode = ''
            for i, (code_key, code_value) in enumerate(tested_code_map.items()):
                scode = switch_body_code
                scode = re.sub(self.__COORD_TAG, code_key, scode)
                scode = re.sub(self.__TCODE_TAG, code_value, scode)
                tcode += '\n'
                tcode += '  case (%s)\n' % i
                tcode += '    \n' + scode + '\n\n'
            code = re.sub(self.__SWITCHBODY_TAG, tcode, code)
            
        # insert the sequential code
        else:
            ((coord_key, tcode),) = tested_code_map.items()
            code = re.sub(self.__COORD_TAG, coord_key, code)
            code = re.sub(self.__TCODE_TAG, tcode, code)

        # return the performance-testing code
        return  code
