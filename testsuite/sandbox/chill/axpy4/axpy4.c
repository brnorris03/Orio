void axpy4(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4) {

/*@ begin PerfTuning (
 def build {
   arg build_command = 'gcc';
   arg libs = '-lm -lrt -fopenmp';
 } 
 def performance_counter {
   arg repetitions = 3;
 }
 def performance_params 
 {
    param TF[] = [2,4,8,16,32,64,128,256,512,1024];
 }

# def performance_test_code { 
#  arg skeleton_code_file = 'skeleton.c';  
#}
 def input_params {
   param N[] = [10240];
 }
 def input_vars {
   decl dynamic double x1[N] = random;
   decl dynamic double x2[N] = random;
   decl dynamic double x3[N] = random;
   decl dynamic double x4[N] = random;
   decl dynamic double y[N] = 0;
   decl double a1 = random;
   decl double a2 = random;
   decl double a3 = random;
   decl double a4 = random;
 }
 def search {
   arg algorithm = 'Exhaustive';
 }
 #  import spec Axpy4TuningSpec; 
) @*/

/*@ begin CHiLL (
   
	tile_by_index(0,{"i"},{TF},{l1_control="ii"},{"ii","i"})CU=1
	cudaize(0,"mm_GPU",{y=10240,x1=10240,x2=10240,x3-10240,x4=10240},{block={"ii","jj"}, thread={"i","j"}},{})CU=2

  ) @*/

int i;
for (i=0; i<N; i++)
  y[i] = y[i] + a1*x1[i] + a2*x2[i] + a3*x3[i] + a4*x4[i];
/*@ end @*/   // CHiLL

/*@ end @*/   // PerfTuning

}
