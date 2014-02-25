void mm(double *a,double *b, double *c) {

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
    param TF[] = [2,4,8,16,32];
    param TF2[] = [2,4,8,16,32];

 }

# def performance_test_code { 
#  arg skeleton_code_file = 'skeleton.c';  
#}
 def input_params {
   param N[] = [1024];
 }
 def input_vars {
   decl dynamic double a[N*N] = random;
   decl dynamic double b[N*N] = random;
   decl dynamic double c[N*N] = random;
#   arg init_file = 'rose__orio_chill_.c';
 }
 def search {
   arg algorithm = 'Exhaustive';
 }
 #  import spec Axpy4TuningSpec; 
) @*/

/*@ begin CHiLL (
   
	tile_by_index(0,{"i"},{TF},{l1_control="ii"},{"ii","i","j","k"})CU=1
	tile_by_index(0,{"j"},{TF2},{l1_control="jj"},{"ii","jj","i","j","k"})CU=2
	cudaize(0,"mm_GPU",{a=1024,b=1024,c=1024},{block={"ii","jj"}, thread={"i","j"}},{})CU=3

  ) @*/

int i,j,k;
for (i=0; i<N; i++){
	for(j=0;j<N;j++){
		for(k=0;k<N;k++){
		  c[i*N + j] = c[i*N +j] + a[i*N + k] * b[k*N + j];
		}
	}
}
/*@ end @*/   // CHiLL

/*@ end @*/   // PerfTuning

}
