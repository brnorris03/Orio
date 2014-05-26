void c_std_d1_1(double *T3, double *T2i, double *v2){


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
    param UF[] = [2,4];

 }
 def input_params {
   param tilesize[] = [16];
   param tile2[] = [16*16];
   param tile4[] = [16*16*16*16];
   param tile6[] = [16*16*16*16*16*16];
 }
 def input_vars {
   decl dynamic double T3[tile6] = random;
   decl dynamic double T2i[tile4] = random;
   decl dynamic double v2[tile4] = random;
 }
 def search {
   arg algorithm = 'Exhaustive';
 }
) @*/

/*@ begin CHiLL (
   

	stm(0,("p4","p5","p6","h1","h2","h3","h7"),"T3")

	permute(0,("p5","p4","p6","h1","h2","h3","h7"))
	cuda(0,block={"p6","h1"},thread={"h3","h2"})
	registers(0,"h7")
	unroll(0,"h7",UF)
	

  ) @*/



	int p4, p5, p6, h1, h2, h3, h7;

	for(p4 = 0; p4 < tilesize; p4++){
		for(p5 = 0; p5 < tilesize; p5++){
			for(p6 = 0; p6 < tilesize; p6++){
				for(h1 = 0; h1 < tilesize; h1++){
					for(h2 = 0; h2 < tilesize; h2++){
						for(h3 = 0; h3 < tilesize; h3++){
							for(h7 = 0; h7 < tilesize; h7++){
								T3[h3 + tilesize*(h2 + tilesize*(h1 + tilesize*(p6 + tilesize*(p5 + tilesize*(p4)))))]  = T3[h3 + tilesize*(h2 + tilesize*(h1 + tilesize*(p6 + tilesize*(p5 + tilesize*(p4)))))] -  (T2i[h7 + tilesize*(p4 + tilesize*(p5 + tilesize*(h1)))] * v2[h3 + tilesize*(h2 + tilesize*(p6 + tilesize*(h7)))]);
							}
						}
					}
				}
			}
		}
	}

/*@ end @*/   // CHiLL

/*@ end @*/   // PerfTuning


}
