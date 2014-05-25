void mm(double *a, double *b, double *c){


/*@ begin PerfTuning (
 def build {
   arg build_command = 'g++';
   arg libs = 'rose__orio_chill_.o -I/usr/local/cuda-5.5/include -L/usr/local/cuda-5.5/lib64 -lcudart -lcuda -lm -lrt';
 } 
 def performance_counter {
   arg repetitions = 3;
 }
 def performance_params 
 {
    param TF[] = [2,4];
    param TF2[] = [2,4];
    param TF3[] = [2,4];

 }
 def input_params {
   param N[] = [1024];
 }
 def input_vars {
   decl dynamic double a[N*N] = random;
   decl dynamic double b[N*N] = random;
   decl dynamic double c[N*N] = random;
 }
 def search {
   arg algorithm = 'Exhaustive';
 }
) @*/

/*@ begin CHiLL (
   

	stm(0,("i","j","k"),"c")

	tile(0,"i",TF,"ii")
	tile(0,"j",TF2,"jj")
	tile(0,"k",TF3,"kk")

	cuda(0,block={"ii","i"},thread={"j","jj"})
	registers(0,"kk")
	

  ) @*/


	int j, i, k;

	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			for(k = 0; k < N; k++){
				c[i*N + j]  = c[i*N + j] +  (a[i*N + k] * b[k*N + j]);
				
			}
		}

	}

/*@ end @*/   // CHiLL

/*@ end @*/   // PerfTuning


}
