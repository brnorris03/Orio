void tensor(
double *u, double *D, double *Dt, double *ur, double *us, double *ut){


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
    param TF[] = [2,5];
#    param TF2[] = [2,5];
#    param TF3[] = [2,5,10,20,25,50];

 }
 def input_params {
   param N[] = [10];
   param nelt[] = [100];
 }
 def input_vars {
   decl dynamic double u[nelt*N*N*N] = random;
   decl dynamic double ur[nelt*N*N*N] = random;
   decl dynamic double us[nelt*N*N*N] = random;
   decl dynamic double ut[nelt*N*N*N] = random;
   decl dynamic double D[N*N] = random;
   decl dynamic double Dt[N*N] = random;
 }
 def search {
   arg algorithm = 'Exhaustive';
 }
) @*/

/*@ begin CHiLL (
   

  	#define M_tinyS 10
	#define K_tinyS 10
	#define N_small 100

	#define MATS_ALL 10

	#define M_small 100
	#define N_tinyS 10

	#define nelt 100

	N = 10
	nelt = 100

	distribute({0,1,2},1)

	tile_by_index(0,{"i"},{TF},{l1_control="ii"},{"l","j","ii","i","k"})CU=1
	
	cudaize(0,"tensor1_GPU",{u=nelt*N*N*N,ur=nelt*N*N*N,us=nelt*N*N*N,ut=nelt*N*N*N,D=N*N,Dt=N*N},{block={"l","j"}, thread={"ii","i"}},{})CU=3
	cudaize(1,"tensor2_GPU",{u=nelt*N*N*N,ur=nelt*N*N*N,us=nelt*N*N*N,ut=nelt*N*N*N,D=N*N,Dt=N*N},{block={"l","j"}, thread={"i","k"}},{})CU=4
	cudaize(2,"tensor3_GPU",{u=nelt*N*N*N,ur=nelt*N*N*N,us=nelt*N*N*N,ut=nelt*N*N*N,D=N*N,Dt=N*N},{block={"l","j"}, thread={"i"}},{})CU=5

	run clean ur us ut 1

  ) @*/



	int i, j, k, l, m;


	for(l =0; l<nelt;l++){
		for(j =0; j<N_small;j++){
			for(i=0; i<M_tinyS;i++){
				for(k =0; k<K_tinyS;k++){

					ur[l*N_small*M_tinyS + j*M_tinyS + i] = ur[l*N_small*M_tinyS + j*M_tinyS + i] + D[k*K_tinyS + i] * u[l*N_small*M_tinyS + j*M_tinyS + k];

				}
			}
		}

		for(j =0; j<MATS_ALL;j++){
			for(i =0; i<N_tinyS;i++){
				for(k=0;k<M_tinyS;k++){
					for(m =0; m<K_tinyS;m++){

						us[l*MATS_ALL*M_tinyS*N_tinyS + j*M_tinyS*N_tinyS + k*N_tinyS + i] = us[l*MATS_ALL*M_tinyS*N_tinyS + j*M_tinyS*N_tinyS + k*N_tinyS + i] + u[l*MATS_ALL*M_tinyS*N_tinyS + j*M_tinyS*K_tinyS + m*K_tinyS + i] * Dt[k*N_tinyS + m];

					}
				}
			}
		}

		for(j =0; j<N_tinyS;j++){
			for(i=0; i<M_small;i++){
				for(k =0; k<K_tinyS;k++){

					ut[l*N_tinyS*M_small + j*M_small + i] = ut[l*N_tinyS*M_small + j*M_small + i] + u[l*N_tinyS*M_small + k*M_small + i] * Dt[j*K_tinyS + k];

				}
			}
		}
	}

/*@ end @*/   // CHiLL

/*@ end @*/   // PerfTuning


}
