
void matmulnaive(double **m1, double **m2,double **mult,int row, int col, char *recvbuff, char *sendbuff)
{

  /*@ begin PerfTuning
  (

   def build {                                                                        
      arg build_command = 'mpicc -O3';      
      arg libs = '';
   }
                 
   def performance_counter {                                                          
      arg method = 'basic timer';
      arg repetitions = 3;                                                           
   }                      

   def performance_params {
       param PRCL[] = ['rendezvous','eager'];
       param MSG[] = [2000];
       param COMM[] = [1,2,3];
   }

   def input_params {
      param row[] = [80];
      param col[] = [80];
      param maxnumcomm = 8;
      param datatype = 'MPI_BYTE';
      param maxmsgsize = 2000; 
   } 

   def input_vars {
      decl dynamic double m1[row*col] = random;
      decl dynamic double m2[row*col] = random;
      decl dynamic double mult[row*col] = random;
      decl dynamic char recvbuff[maxnumcomm][maxmsgsize] = random;
      decl dynamic char sendbuff[maxnumcomm][maxmsgsize] = random;
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
