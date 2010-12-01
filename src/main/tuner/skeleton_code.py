#
# The skeleton code used for performance testing
#

import os, re, sys
from orio.main.util.globals import *

#-----------------------------------------------------
SEQ_TIMER = '''
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define BIG_NUMBER 147483647.0

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
#ifdef __GNUC__
double getClock()
{
    long sec;
    double secx;
    struct tms realbuf;

    times(&realbuf);
    secx = ( realbuf.tms_stime + realbuf.tms_utime ) / (float) CLOCKS_PER_SEC;
    return ((double) secx);
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
#endif
'''

SEQ_DEFAULT = r'''
/*@ global @*/

int main(int argc, char *argv[])
{
  /*@ prologue @*/

  double orio_t_start, orio_t_end, orio_t, orio_t_total=0, orio_t_min = BIG_NUMBER;
  int orio_i;

  for (orio_i=0; orio_i<REPS; orio_i++)
  {
    orio_t_start = getClock();
    
    /*@ tested code @*/

    orio_t_end = getClock();
    orio_t = orio_t_end - orio_t_start;
    if (orio_t < orio_t_min) orio_t_min = orio_t;
  }
  orio_t_total = orio_t_min;
  
  printf("{'/*@ coordinate @*/' : %g}", orio_t_total);

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
      for (orio_i=0; orio_i<REPS; orio_i++)
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
    do orio_i = 0, REPS-1
    
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
    
        do orio_i = 1, REPS
    
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
            err('main.tuner.skeleton_code:  missing "global" tag in the skeleton code')

        match_obj = re.search(self.__PROLOGUE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code:  missing "prologue" tag in the skeleton code')

        match_obj = re.search(self.__EPILOGUE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code:  missing "epilogue" tag in the skeleton code')

        match_obj = re.search(self.__TCODE_TAG, code)
        if not match_obj:
            err('main.tuner.skeleton_code:  missing "tested code" tag in the skeleton code')

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
            
            match_obj = re.search(self.__COORD_TAG, switch_body_code)
            if not match_obj:
                err('main.tuner.skeleton_code:  missing "coordinate" tag in the switch body statement')

    #-----------------------------------------------------

    def insertCode(self, global_code, prologue_code, epilogue_code, tested_code_map):
        '''
        Insert code fragments into the skeleton driver code.
        
        @return: Complete specialized C source code string for the performance testing driver.
        
        @param global_code:
        @param prologue_code: 
        @param epilogue code:
        @param tested_code_map:
        '''

        # check the given tested code mapping
        if len(tested_code_map) == 0:
            err('main.tuner.skeleton_code internal error:  the number of tested codes cannot be zero')
        if not self.use_parallel_search and len(tested_code_map) != 1:
            err('main.tuner.skeleton_code internal error:  the number of tested sequential codes must be exactly one')

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
            ((coord_key, tcode),) = tested_code_map.items()
            code = re.sub(self.__COORD_TAG, coord_key, code)
            code = re.sub(self.__TCODE_TAG, tcode, code)

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
            print 'error: missing "prologue" tag in the skeleton code'
            sys.exit(1)

        match_obj = re.search(self.__EPILOGUE_TAG, code)
        if not match_obj:
            err('missing "epilogue" tag in the skeleton code')

        match_obj = re.search(self.__TCODE_TAG, code)
        if not match_obj:
            err('missing "tested code" tag in the skeleton code')

        match_obj = re.search(self.__COORD_TAG, code)
        if not match_obj:
            err('missing "coordinate" tag in the skeleton code')
            
        if self.use_parallel_search:

            match_obj = re.search(self.__BEGIN_SWITCHBODY_TAG, code)
            if not match_obj:
                err('error: missing "begin switch body" tag in the skeleton code')
        
            match_obj = re.search(self.__END_SWITCHBODY_TAG, code)
            if not match_obj:
                err('missing "end switch body" tag in the skeleton code')
        
            match_obj = re.search(self.__SWITCHBODY_TAG, code)
            if not match_obj:
                err('internal error: missing placement of switch body statement')

            switch_body_code = match_obj.group(1)

            match_obj = re.search(self.__TCODE_TAG, switch_body_code)
            if not match_obj:
                err('missing "tested code" tag in the switch body statement')
            
            match_obj = re.search(self.__COORD_TAG, switch_body_code)
            if not match_obj:
                err('missing "coordinate" tag in the switch body statement')

    #-----------------------------------------------------

    def insertCode(self, decl_code, prologue_code, epilogue_code, tested_code_map):
        '''To insert code fragments into the skeleton code'''

        # check the given tested code mapping
        if len(tested_code_map) == 0:
            err('internal error: the number of tested codes cannot be zero')

        if not self.use_parallel_search and len(tested_code_map) != 1:
            err('internal error: the number of tested sequential codes must be exactly one')

        # initialize the performance-testing code
        code = self.code

        # insert global definitions, prologue, and epilogue codes
        code = re.sub(self.__DECLARATIONS_TAG, decl_code, code)
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
