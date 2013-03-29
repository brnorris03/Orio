
void matmulnaive(double **m1, double **m2,double **mult,int row, int col, char *recvbuff, char *sendbuff, int loopiters)
{

  /*@ begin PerfTuning
  (

   def build {                                                                        
      arg build_command = 'mpicc -O3 -qarch=auto -qsmp=omp:noauto';      
      arg libs = '';
      arg batch_command = 'qsub -n 1 --proccount 8 -t 5 --mode c8';
      arg status_command = 'qstat';                                                  
      arg num_procs = 8; 
   }
                 
   def performance_counter {                                                          
      arg method = 'bgp counter';
      arg repetitions = 1;                                                           
   }                      

   def performance_params {
       param PRCL[] = ['rendezvous','eager'];
       param MSG[] = [200];
       param COMM[] = [1];
   }

   def input_params {
      param row[] = [80];
      param col[] = [80];
      param maxnumcomm = 8;
      param datatype = 'MPI_BYTE';
      param maxmsgsize = 200; 
   } 

   def input_vars {
      decl dynamic double m1[row*col] = random;
      decl dynamic double m2[row*col] = random;
      decl dynamic double mult[row*col] = random;
      decl dynamic char recvbuff[maxnumcomm][maxmsgsize] = random;
      decl dynamic char sendbuff[maxnumcomm][maxmsgsize] = random;
      decl int loopiters = 0;
   }

   def search {
     arg algorithm = 'Exhaustive';
   }

  ) @*/

  int i,j,k;

  /*@ begin Loop
  (
  transform MPIOverlap(protocol=PRCL,msgsize=MSG,communication=COMM)
  for(i=0;i<row;i++)
    {
      for(j=0;j<col;j++)
        {
          for(k=0;k<row;k++)
            {
              mult[i*row+j] += m1[i*row+k] * m2[k*row+j];
            }
        }
    }
  ) @*/


  for(i=0;i<row;i++)
    {
      for(j=0;j<col;j++)
        {
          for(k=0;k<row;k++)
            {
              mult[i*row+j] += m1[i*row+k] * m2[k*row+j];
            }
        }
    }

  /*@ end @*/
  /*@ end @*/
}
