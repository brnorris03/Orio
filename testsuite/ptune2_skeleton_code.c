#include <stdio.h> 
#include <stdlib.h>
#include <sys/time.h>  

/*@ global @*/ 

double rtclock()   
{   
  struct timezone tzp; 
  struct timeval tp;   
  int stat;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6); 
}   

int main() 
{   
  /*@ prologue @*/ 

  double annot_t_start=0, annot_t_end=0, annot_t_total=0;
  int annot_i; 

  for (annot_i=0; annot_i<REPS; annot_i++)
{ 
  annot_t_start = rtclock();

  /*@ tested code @*/

  annot_t_end = rtclock();  
  annot_t_total += annot_t_end - annot_t_start;
} 

  annot_t_total = annot_t_total / REPS;   
  printf("%f\n", annot_t_total);  

  /*@ epilogue @*/ 

  return 0;
}  
