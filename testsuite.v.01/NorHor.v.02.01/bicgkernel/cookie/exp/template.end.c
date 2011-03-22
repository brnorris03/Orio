


      annot_t_end = rtclock();
      annot_t_total += annot_t_end - annot_t_start;
    }

  annot_t_total = annot_t_total / REPS;

#ifndef TEST
  printf("%f\n", annot_t_total);
#else
  {
    int i, j;
    for (i=0; i<ny; i++) {
      if (i%100==0)
        printf("\n");
      printf("%f ",s[i]);
    }
    printf("\n");
    for (i=0; i<nx; i++) {
      if (i%100==0)
        printf("\n");
      printf("%f ",q[i]);
    }
  }
#endif

  return ((int) (s[0]+q[0]));

}

