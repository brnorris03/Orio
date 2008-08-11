
void axpy5(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4, double a5, double *x5) {
    
#pragma disjoint (*x1,*x2,*x3,*x4,*x5,*y) 

    register int i;

    // Small problem size
    if (n <= 600) {
	
	// parallelize=False, ufactor=4
	if ((((int)(x1)|(int)(x2)|(int)(x3)|(int)(x4)|(int)(x5)|(int)(y)) & 0xF) == 0) {
	    __alignx(16,x1); 
	    __alignx(16,x2); 
	    __alignx(16,x3); 
	    __alignx(16,x4); 
	    __alignx(16,x5); 
	    __alignx(16,y); 
	    for (i=0; i<=n-4; i=i+4) {
		y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
		y[(i+1)]=y[(i+1)]+a1*x1[(i+1)]+a2*x2[(i+1)]+a3*x3[(i+1)]+a4*x4[(i+1)]+a5*x5[(i+1)];
		y[(i+2)]=y[(i+2)]+a1*x1[(i+2)]+a2*x2[(i+2)]+a3*x3[(i+2)]+a4*x4[(i+2)]+a5*x5[(i+2)];
		y[(i+3)]=y[(i+3)]+a1*x1[(i+3)]+a2*x2[(i+3)]+a3*x3[(i+3)]+a4*x4[(i+3)]+a5*x5[(i+3)];
	    }
	    for (; i<=n-1; i=i+1) 
		y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
  
	} else {

	    for (i=0; i<=n-4; i=i+4) {
		y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
		y[(i+1)]=y[(i+1)]+a1*x1[(i+1)]+a2*x2[(i+1)]+a3*x3[(i+1)]+a4*x4[(i+1)]+a5*x5[(i+1)];
		y[(i+2)]=y[(i+2)]+a1*x1[(i+2)]+a2*x2[(i+2)]+a3*x3[(i+2)]+a4*x4[(i+2)]+a5*x5[(i+2)];
		y[(i+3)]=y[(i+3)]+a1*x1[(i+3)]+a2*x2[(i+3)]+a3*x3[(i+3)]+a4*x4[(i+3)]+a5*x5[(i+3)];
	    }
	    for (; i<=n-1; i=i+1) 
		y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
	}
	
    }
    
    // Large problem size
    else {
	
	// parallelize=False, ufactor=8
	if ((((int)(x1)|(int)(x2)|(int)(x3)|(int)(x4)|(int)(x5)|(int)(y)) & 0xF) == 0) {
	    __alignx(16,x1); 
	    __alignx(16,x2); 
	    __alignx(16,x3); 
	    __alignx(16,x4); 
	    __alignx(16,x5); 
	    __alignx(16,y); 

	    for (i=0; i<=n-8; i=i+8) {
		y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
		y[(i+1)]=y[(i+1)]+a1*x1[(i+1)]+a2*x2[(i+1)]+a3*x3[(i+1)]+a4*x4[(i+1)]+a5*x5[(i+1)];
		y[(i+2)]=y[(i+2)]+a1*x1[(i+2)]+a2*x2[(i+2)]+a3*x3[(i+2)]+a4*x4[(i+2)]+a5*x5[(i+2)];
		y[(i+3)]=y[(i+3)]+a1*x1[(i+3)]+a2*x2[(i+3)]+a3*x3[(i+3)]+a4*x4[(i+3)]+a5*x5[(i+3)];
		y[(i+4)]=y[(i+4)]+a1*x1[(i+4)]+a2*x2[(i+4)]+a3*x3[(i+4)]+a4*x4[(i+4)]+a5*x5[(i+4)];
		y[(i+5)]=y[(i+5)]+a1*x1[(i+5)]+a2*x2[(i+5)]+a3*x3[(i+5)]+a4*x4[(i+5)]+a5*x5[(i+5)];
		y[(i+6)]=y[(i+6)]+a1*x1[(i+6)]+a2*x2[(i+6)]+a3*x3[(i+6)]+a4*x4[(i+6)]+a5*x5[(i+6)];
		y[(i+7)]=y[(i+7)]+a1*x1[(i+7)]+a2*x2[(i+7)]+a3*x3[(i+7)]+a4*x4[(i+7)]+a5*x5[(i+7)];
	    }
	    for (; i<=n-1; i=i+1) 
		y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
	    
	} else {
	    
	    for (i=0; i<=n-8; i=i+8) {
		y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
		y[(i+1)]=y[(i+1)]+a1*x1[(i+1)]+a2*x2[(i+1)]+a3*x3[(i+1)]+a4*x4[(i+1)]+a5*x5[(i+1)];
		y[(i+2)]=y[(i+2)]+a1*x1[(i+2)]+a2*x2[(i+2)]+a3*x3[(i+2)]+a4*x4[(i+2)]+a5*x5[(i+2)];
		y[(i+3)]=y[(i+3)]+a1*x1[(i+3)]+a2*x2[(i+3)]+a3*x3[(i+3)]+a4*x4[(i+3)]+a5*x5[(i+3)];
		y[(i+4)]=y[(i+4)]+a1*x1[(i+4)]+a2*x2[(i+4)]+a3*x3[(i+4)]+a4*x4[(i+4)]+a5*x5[(i+4)];
		y[(i+5)]=y[(i+5)]+a1*x1[(i+5)]+a2*x2[(i+5)]+a3*x3[(i+5)]+a4*x4[(i+5)]+a5*x5[(i+5)];
		y[(i+6)]=y[(i+6)]+a1*x1[(i+6)]+a2*x2[(i+6)]+a3*x3[(i+6)]+a4*x4[(i+6)]+a5*x5[(i+6)];
		y[(i+7)]=y[(i+7)]+a1*x1[(i+7)]+a2*x2[(i+7)]+a3*x3[(i+7)]+a4*x4[(i+7)]+a5*x5[(i+7)];
	    }
	    for (; i<=n-1; i=i+1) 
		y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
	}
    }
}
