



      annot_t_end = rtclock();
      annot_t_total += annot_t_end - annot_t_start;
    }

  annot_t_total = annot_t_total / REPS;

#ifndef TEST
  printf("%f\n", annot_t_total);
#else
  {
    int i, j;
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        if (j%100==0)
          printf("\n");
        printf("%f ",A[i][j]);
      }
      printf("\n");
    }
  }
#endif

  return ((int) A[0][0]);

}
