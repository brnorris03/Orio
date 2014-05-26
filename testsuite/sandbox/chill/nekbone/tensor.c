void tensor(double *u, double *D, double *Dt, double *ur, double *us, double *ut){


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
   param UF[] = [2,5];
 }
 def input_params {
   param lx[] = [10];
   param ly[] = [10];
   param lz[] = [10];
   param nelt[] = [100];
 }
 def input_vars {
   decl dynamic double u[nelt*lx*ly*lz] = random;
   decl dynamic double ur[nelt*lx*ly*lz] = random;
   decl dynamic double us[nelt*lx*ly*lz] = random;
   decl dynamic double ut[nelt*lx*ly*lz] = random;
   decl dynamic double D[lx*ly] = random;
   decl dynamic double Dt[lx*ly] = random;
 }
 def search {
   arg algorithm = 'Exhaustive';
 }
) @*/

/*@ begin CHiLL (
   

	stm(0,("e","j","i","k"),"ur")
	stm(1,("e","j","i","k","m"),"us")
	stm(2,("e","j","i","k"),"ut")

	distribute(1)

	tile(0,"i",TF,"ii")
	tile(2,"i",2,"ii")

	cuda(0,block={"e","j"},thread={"ii","i"})
	registers(0,"k")
	unroll(0,"k",UF)

	cuda(1,block={"e","j"}, thread={"i","k"})
	registers(1,"m")
	unroll(1,"m",2)
		
	cuda(2,block={"e","j"}, thread={"ii","i"})
	registers(2,"k")
	unroll(2,"k",2)


  ) @*/


	int e, j, i, k, m;

	for(e = 0; e < nelt; e++){
		for(j = 0; j < lx*ly; j++){
			for(i = 0; i < lz; i++){
				for(k = 0; k < lx; k++){
					ur[e*lx*ly*lz + j*lz + i]  = ur[e*lx*ly*lz + j*lz + i] +  (D[k*ly + i] * u[e*lx*ly*lz + j*lz + k]);
				}
			}
		}
	
		for(j = 0; j < lx; j++){
			for(i = 0; i < ly; i++){
				for(k = 0; k < lx; k++){
					for(m = 0; m < ly; m++){
						us[e*lx*ly*lx + j*ly*lx + i*lx + k]  = us[e*lx*ly*lx + j*ly*lx + i*lx + k] +  (u[e*lx*ly*lx + j*ly*lx + m*lx + k] * Dt[i*ly + m]);
					}
				}
			}
		}
	
		for(j = 0; j < lx; j++){
			for(i = 0; i < ly*lz; i++){
				for(k = 0; k < lx; k++){
					ut[e*lx*ly*lz + j*ly*lz + i]  = ut[e*lx*ly*lz + j*ly*lz + i] +  (u[e*lx*ly*lz + k*ly*lz + i] * Dt[j*ly + k]);
				}
			}
		}
	}

/*@ end @*/   // CHiLL

/*@ end @*/   // PerfTuning


}
