


    annot_t_end = rtclock();
    annot_t_total += annot_t_end - annot_t_start;
  }
  
  annot_t_total = annot_t_total / REPS;

#ifndef TEST
  printf("%f\n", annot_t_total);
#else
  {
    int i,j;
    for (i=0; i<nx; i++) {
      for (j=0; j<ny; j++)  {
	if (j%100==0)
          printf("\n");
        printf("%f ",hz[i][j]);
      }
      printf("\n");
    }
  }
#endif


  return ((int) hz[0][0]); 

}
                                    
