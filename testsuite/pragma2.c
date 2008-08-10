
/*@ begin Pragma('omp parallel for') @*/   
for (i=0; i<n; i++)
  y[i] = b[i] + ss*a[i];
/*@ end @*/


