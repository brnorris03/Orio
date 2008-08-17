
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define tmax T
#define nx N
#define ny N
double ex[nx][ny +1];
double ey[nx +1][ny];
double hz[nx][ny];

void init_arrays()
{
    int i, j;
    for (i=0; i<nx+1; i++)  {
        for (j=0; j<ny; j++)  {
            ey[i][j] = 0;
        }
    }
    for (i=0; i<nx; i++)  {
        for (j=0; j<ny+1; j++)  {
            ex[i][j] = 0;
        }
    }
    for (j=0; j<ny; j++)  {
        ey[0][j] = ((double)j)/ny;
    }
    for (i=0; i<nx; i++)    {
        for (j=0; j<ny; j++)  {
            hz[i][j] = 0;
        }
    }
}

double rtclock()
{
  struct timezone tzp;
  struct timeval tp;
  int stat;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}

int main()
{
  init_arrays();

  double annot_t_start=0, annot_t_end=0, annot_t_total=0;
  int annot_i;

  for (annot_i=0; annot_i<REPS; annot_i++)
  {
    annot_t_start = rtclock();
    
#include <math.h>
#include <assert.h>
#include <omp.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))





 int c1, c2, c3, c4, c5, c6;
 register int lb, ub, lb1, ub1, lb2, ub2;
 register int lbv, ubv;

for (c1=-1;c1<=floord(2*tmax+ny-2,32);c1++) {
 lb1=max(max(ceild(32*c1-tmax+1,32),0),ceild(32*c1-31,64));
 ub1=min(min(floord(32*c1+31,32),floord(32*c1+ny+31,64)),floord(tmax+ny-1,32));
#pragma omp parallel for shared(c1,lb1,ub1) private(c2,c3,c4,c5,c6)
 for (c2=lb1; c2<=ub1; c2++) {
    for (c3=max(max(max(max(max(max(max(max(max(max(ceild(32*c2-ny-30,32),0),ceild(32*c1-32*c2-31*ny-899,992)),ceild(64*c1-96*c2-61,32)),ceild(1024*c1-2016*c2-30*nx-931,32)),ceild(992*c1-1952*c2-30*nx-ny-899,32)),ceild(32*c1-1024*c2-30*nx-931,32)),ceild(32*c1-32*c2-ny-29,32)),ceild(32*c1-32*c2-31,32)),ceild(32*c1-1024*c2-1891,992)),ceild(32*c1-992*c2-30*nx-ny-899,32));c3<=min(min(min(floord(tmax+nx-1,32),floord(32*c1-32*c2+nx+31,32)),floord(32*c2+nx+30,32)),c1+31*c2+nx+30);c3++) {
      if ((c1 <= floord(32*c2+32*c3-nx,32)) && (c2 <= floord(32*c3-nx+ny,32)) && (c3 >= ceild(nx,32))) {
        for (c5=max(32*c3-nx+1,32*c2);c5<=min(32*c3-nx+ny,32*c2+31);c5++) {
          {hz[nx-1][-32*c3+c5+nx-1]=hz[nx-1][-32*c3+c5+nx-1]-((double)(7))/10*(ey[1+nx-1][-32*c3+c5+nx-1]+ex[nx-1][1+-32*c3+c5+nx-1]-ex[nx-1][-32*c3+c5+nx-1]-ey[nx-1][-32*c3+c5+nx-1]);} ;
        }
      }
      if ((c1 <= floord(64*c2-ny,32)) && (c2 >= max(ceild(ny,32),ceild(32*c3-nx+ny+1,32)))) {
        for (c6=max(32*c2-ny+1,32*c3);c6<=min(32*c2+nx-ny,32*c3+31);c6++) {
          {hz[-32*c2+c6+ny-1][ny-1]=hz[-32*c2+c6+ny-1][ny-1]-((double)(7))/10*(ey[1+-32*c2+c6+ny-1][ny-1]+ex[-32*c2+c6+ny-1][1+ny-1]-ex[-32*c2+c6+ny-1][ny-1]-ey[-32*c2+c6+ny-1][ny-1]);} ;
        }
      }
      if ((c1 == c2+c3) && (nx >= 2)) {
        for (c4=max(max(0,32*c3),32*c2-ny+1);c4<=min(min(min(32*c2-1,32*c3-nx+31),tmax-1),32*c2-ny+31);c4++) {
          for (c5=32*c2;c5<=c4+ny-1;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            for (c6=c4+1;c6<=c4+nx-1;c6++) {
              {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
              {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
              {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
            }
            {hz[nx-1][-c4+c5-1]=hz[nx-1][-c4+c5-1]-((double)(7))/10*(ey[1+nx-1][-c4+c5-1]+ex[nx-1][1+-c4+c5-1]-ex[nx-1][-c4+c5-1]-ey[nx-1][-c4+c5-1]);} ;
          }
          for (c6=c4+1;c6<=c4+nx;c6++) {
            {hz[-c4+c6-1][ny-1]=hz[-c4+c6-1][ny-1]-((double)(7))/10*(ey[1+-c4+c6-1][ny-1]+ex[-c4+c6-1][1+ny-1]-ex[-c4+c6-1][ny-1]-ey[-c4+c6-1][ny-1]);} ;
          }
        }
      }
      if ((c1 == c2+c3) && (nx >= 2) && (ny >= 2)) {
        for (c4=max(max(32*c2,0),32*c3);c4<=min(min(32*c3-nx+31,tmax-1),32*c2-ny+31);c4++) {
          {ey[0][0]=c4;} ;
          for (c6=c4+1;c6<=c4+nx-1;c6++) {
            {ey[-c4+c6][0]=ey[-c4+c6][0]-((double)(1))/2*(hz[-c4+c6][0]-hz[-c4+c6-1][0]);} ;
          }
          for (c5=c4+1;c5<=c4+ny-1;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            for (c6=c4+1;c6<=c4+nx-1;c6++) {
              {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
              {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
              {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
            }
            {hz[nx-1][-c4+c5-1]=hz[nx-1][-c4+c5-1]-((double)(7))/10*(ey[1+nx-1][-c4+c5-1]+ex[nx-1][1+-c4+c5-1]-ex[nx-1][-c4+c5-1]-ey[nx-1][-c4+c5-1]);} ;
          }
          for (c6=c4+1;c6<=c4+nx;c6++) {
            {hz[-c4+c6-1][ny-1]=hz[-c4+c6-1][ny-1]-((double)(7))/10*(ey[1+-c4+c6-1][ny-1]+ex[-c4+c6-1][1+ny-1]-ex[-c4+c6-1][ny-1]-ey[-c4+c6-1][ny-1]);} ;
          }
        }
      }
      if ((c1 == c2+c3) && (nx >= 2)) {
        for (c4=max(max(0,32*c3),32*c2-ny+32);c4<=min(min(tmax-1,32*c2-1),32*c3-nx+31);c4++) {
          for (c5=32*c2;c5<=32*c2+31;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            for (c6=c4+1;c6<=c4+nx-1;c6++) {
              {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
              {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
              {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
            }
            {hz[nx-1][-c4+c5-1]=hz[nx-1][-c4+c5-1]-((double)(7))/10*(ey[1+nx-1][-c4+c5-1]+ex[nx-1][1+-c4+c5-1]-ex[nx-1][-c4+c5-1]-ey[nx-1][-c4+c5-1]);} ;
          }
        }
      }
      if (c1 == c2+c3) {
        for (c4=max(max(max(32*c3-nx+32,0),32*c3),32*c2-ny+1);c4<=min(min(min(32*c2-1,tmax-1),32*c2-ny+31),32*c3+30);c4++) {
          for (c5=32*c2;c5<=c4+ny-1;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            for (c6=c4+1;c6<=32*c3+31;c6++) {
              {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
              {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
              {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
            }
          }
          for (c6=c4+1;c6<=32*c3+31;c6++) {
            {hz[-c4+c6-1][ny-1]=hz[-c4+c6-1][ny-1]-((double)(7))/10*(ey[1+-c4+c6-1][ny-1]+ex[-c4+c6-1][1+ny-1]-ex[-c4+c6-1][ny-1]-ey[-c4+c6-1][ny-1]);} ;
          }
        }
      }
      if ((c1 == c2+c3) && (nx >= 2)) {
        for (c4=max(max(max(32*c2,0),32*c3),32*c2-ny+32);c4<=min(min(tmax-1,32*c2+30),32*c3-nx+31);c4++) {
          {ey[0][0]=c4;} ;
          for (c6=c4+1;c6<=c4+nx-1;c6++) {
            {ey[-c4+c6][0]=ey[-c4+c6][0]-((double)(1))/2*(hz[-c4+c6][0]-hz[-c4+c6-1][0]);} ;
          }
          for (c5=c4+1;c5<=32*c2+31;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            for (c6=c4+1;c6<=c4+nx-1;c6++) {
              {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
              {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
              {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
            }
            {hz[nx-1][-c4+c5-1]=hz[nx-1][-c4+c5-1]-((double)(7))/10*(ey[1+nx-1][-c4+c5-1]+ex[nx-1][1+-c4+c5-1]-ex[nx-1][-c4+c5-1]-ey[nx-1][-c4+c5-1]);} ;
          }
        }
      }
      if ((c1 == c2+c3) && (ny >= 2)) {
        for (c4=max(max(max(32*c2,32*c3-nx+32),0),32*c3);c4<=min(min(tmax-1,32*c2-ny+31),32*c3+30);c4++) {
          {ey[0][0]=c4;} ;
          for (c6=c4+1;c6<=32*c3+31;c6++) {
            {ey[-c4+c6][0]=ey[-c4+c6][0]-((double)(1))/2*(hz[-c4+c6][0]-hz[-c4+c6-1][0]);} ;
          }
          for (c5=c4+1;c5<=c4+ny-1;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            for (c6=c4+1;c6<=32*c3+31;c6++) {
              {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
              {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
              {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
            }
          }
          for (c6=c4+1;c6<=32*c3+31;c6++) {
            {hz[-c4+c6-1][ny-1]=hz[-c4+c6-1][ny-1]-((double)(7))/10*(ey[1+-c4+c6-1][ny-1]+ex[-c4+c6-1][1+ny-1]-ex[-c4+c6-1][ny-1]-ey[-c4+c6-1][ny-1]);} ;
          }
        }
      }
      if (c1 == c2+c3) {
        for (c4=max(max(max(0,32*c3),32*c3-nx+32),32*c2-ny+32);c4<=min(min(tmax-1,32*c3+30),32*c2-1);c4++) {
          for (c5=32*c2;c5<=32*c2+31;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            for (c6=c4+1;c6<=32*c3+31;c6++) {
              {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
              {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
              {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
            }
          }
        }
      }
      if (c1 == c2+c3) {
        for (c4=max(max(max(max(32*c2,32*c3-nx+32),0),32*c3),32*c2-ny+32);c4<=min(min(tmax-1,32*c3+30),32*c2+30);c4++) {
          {ey[0][0]=c4;} ;
          for (c6=c4+1;c6<=32*c3+31;c6++) {
            {ey[-c4+c6][0]=ey[-c4+c6][0]-((double)(1))/2*(hz[-c4+c6][0]-hz[-c4+c6-1][0]);} ;
          }
          for (c5=c4+1;c5<=32*c2+31;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            for (c6=c4+1;c6<=32*c3+31;c6++) {
              {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
              {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
              {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
            }
          }
        }
      }
      if ((c1 == c2+c3) && (nx >= 2) && (ny == 1)) {
        for (c4=max(max(0,32*c3),32*c2);c4<=min(min(tmax-1,32*c2+30),32*c3+30);c4++) {
          {ey[0][0]=c4;} ;
          for (c6=c4+1;c6<=min(c4+nx-1,32*c3+31);c6++) {
            {ey[-c4+c6][0]=ey[-c4+c6][0]-((double)(1))/2*(hz[-c4+c6][0]-hz[-c4+c6-1][0]);} ;
          }
          for (c6=c4+1;c6<=min(32*c3+31,c4+nx);c6++) {
            {hz[-c4+c6-1][0]=hz[-c4+c6-1][0]-((double)(7))/10*(ey[1+-c4+c6-1][0]+ex[-c4+c6-1][1+0]-ex[-c4+c6-1][0]-ey[-c4+c6-1][0]);} ;
          }
        }
      }
      if ((c1 == c2+c3) && (nx == 1)) {
        for (c4=max(max(0,32*c3),32*c2-ny+1);c4<=min(min(min(32*c2-1,tmax-1),32*c2-ny+31),32*c3+30);c4++) {
          for (c5=32*c2;c5<=c4+ny-1;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            {hz[0][-c4+c5-1]=hz[0][-c4+c5-1]-((double)(7))/10*(ey[1+0][-c4+c5-1]+ex[0][1+-c4+c5-1]-ex[0][-c4+c5-1]-ey[0][-c4+c5-1]);} ;
          }
          {hz[0][ny-1]=hz[0][ny-1]-((double)(7))/10*(ey[1+0][ny-1]+ex[0][1+ny-1]-ex[0][ny-1]-ey[0][ny-1]);} ;
        }
      }
      if ((c1 == c2+c3) && (nx == 1) && (ny >= 2)) {
        for (c4=max(max(32*c2,0),32*c3);c4<=min(min(tmax-1,32*c2-ny+31),32*c3+30);c4++) {
          {ey[0][0]=c4;} ;
          for (c5=c4+1;c5<=c4+ny-1;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            {hz[0][-c4+c5-1]=hz[0][-c4+c5-1]-((double)(7))/10*(ey[1+0][-c4+c5-1]+ex[0][1+-c4+c5-1]-ex[0][-c4+c5-1]-ey[0][-c4+c5-1]);} ;
          }
          {hz[0][ny-1]=hz[0][ny-1]-((double)(7))/10*(ey[1+0][ny-1]+ex[0][1+ny-1]-ex[0][ny-1]-ey[0][ny-1]);} ;
        }
      }
      if ((c1 == c2+c3) && (nx == 1)) {
        for (c4=max(max(0,32*c3),32*c2-ny+32);c4<=min(min(tmax-1,32*c3+30),32*c2-1);c4++) {
          for (c5=32*c2;c5<=32*c2+31;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            {hz[0][-c4+c5-1]=hz[0][-c4+c5-1]-((double)(7))/10*(ey[1+0][-c4+c5-1]+ex[0][1+-c4+c5-1]-ex[0][-c4+c5-1]-ey[0][-c4+c5-1]);} ;
          }
        }
      }
      if ((c1 == c2+c3) && (nx == 1)) {
        for (c4=max(max(max(32*c2,0),32*c3),32*c2-ny+32);c4<=min(min(tmax-1,32*c3+30),32*c2+30);c4++) {
          {ey[0][0]=c4;} ;
          for (c5=c4+1;c5<=32*c2+31;c5++) {
            {ey[0][-c4+c5]=c4;} ;
            {ex[0][-c4+c5]=ex[0][-c4+c5]-((double)(1))/2*(hz[0][-c4+c5]-hz[0][-c4+c5-1]);} ;
            {hz[0][-c4+c5-1]=hz[0][-c4+c5-1]-((double)(7))/10*(ey[1+0][-c4+c5-1]+ex[0][1+-c4+c5-1]-ex[0][-c4+c5-1]-ey[0][-c4+c5-1]);} ;
          }
        }
      }
      if ((c1 == c2+c3) && (nx == 1) && (ny == 1)) {
        for (c4=max(max(0,32*c3),32*c2);c4<=min(min(tmax-1,32*c2+30),32*c3+30);c4++) {
          {ey[0][0]=c4;} ;
          {hz[0][0]=hz[0][0]-((double)(7))/10*(ey[1+0][0]+ex[0][1+0]-ex[0][0]-ey[0][0]);} ;
        }
      }
      for (c4=max(max(max(0,32*c3-nx+1),32*c2-ny+1),32*c1-32*c2);c4<=min(min(min(min(min(32*c2-1,32*c3-nx+31),32*c3-1),tmax-1),32*c1-32*c2+31),32*c2-ny+31);c4++) {
        for (c5=32*c2;c5<=c4+ny-1;c5++) {
          for (c6=32*c3;c6<=c4+nx-1;c6++) {
            {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
            {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
            {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
          }
          {hz[nx-1][-c4+c5-1]=hz[nx-1][-c4+c5-1]-((double)(7))/10*(ey[1+nx-1][-c4+c5-1]+ex[nx-1][1+-c4+c5-1]-ex[nx-1][-c4+c5-1]-ey[nx-1][-c4+c5-1]);} ;
        }
        for (c6=32*c3;c6<=c4+nx;c6++) {
          {hz[-c4+c6-1][ny-1]=hz[-c4+c6-1][ny-1]-((double)(7))/10*(ey[1+-c4+c6-1][ny-1]+ex[-c4+c6-1][1+ny-1]-ex[-c4+c6-1][ny-1]-ey[-c4+c6-1][ny-1]);} ;
        }
      }
      if (ny >= 2) {
        for (c4=max(max(max(32*c2,0),32*c3-nx+1),32*c1-32*c2);c4<=min(min(min(min(32*c3-nx+31,32*c3-1),tmax-1),32*c1-32*c2+31),32*c2-ny+31);c4++) {
          for (c6=32*c3;c6<=c4+nx-1;c6++) {
            {ey[-c4+c6][0]=ey[-c4+c6][0]-((double)(1))/2*(hz[-c4+c6][0]-hz[-c4+c6-1][0]);} ;
          }
          for (c5=c4+1;c5<=c4+ny-1;c5++) {
            for (c6=32*c3;c6<=c4+nx-1;c6++) {
              {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
              {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
              {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
            }
            {hz[nx-1][-c4+c5-1]=hz[nx-1][-c4+c5-1]-((double)(7))/10*(ey[1+nx-1][-c4+c5-1]+ex[nx-1][1+-c4+c5-1]-ex[nx-1][-c4+c5-1]-ey[nx-1][-c4+c5-1]);} ;
          }
          for (c6=32*c3;c6<=c4+nx;c6++) {
            {hz[-c4+c6-1][ny-1]=hz[-c4+c6-1][ny-1]-((double)(7))/10*(ey[1+-c4+c6-1][ny-1]+ex[-c4+c6-1][1+ny-1]-ex[-c4+c6-1][ny-1]-ey[-c4+c6-1][ny-1]);} ;
          }
        }
      }
      for (c4=max(max(max(0,32*c1-32*c2),32*c3-nx+1),32*c2-ny+32);c4<=min(min(min(min(tmax-1,32*c3-1),32*c1-32*c2+31),32*c2-1),32*c3-nx+31);c4++) {
        for (c5=32*c2;c5<=32*c2+31;c5++) {
          for (c6=32*c3;c6<=c4+nx-1;c6++) {
            {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
            {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
            {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
          }
          {hz[nx-1][-c4+c5-1]=hz[nx-1][-c4+c5-1]-((double)(7))/10*(ey[1+nx-1][-c4+c5-1]+ex[nx-1][1+-c4+c5-1]-ex[nx-1][-c4+c5-1]-ey[nx-1][-c4+c5-1]);} ;
        }
      }
      for (c4=max(max(max(32*c3-nx+32,0),32*c2-ny+1),32*c1-32*c2);c4<=min(min(min(min(32*c2-1,32*c3-1),tmax-1),32*c1-32*c2+31),32*c2-ny+31);c4++) {
        for (c5=32*c2;c5<=c4+ny-1;c5++) {
          for (c6=32*c3;c6<=32*c3+31;c6++) {
            {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
            {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
            {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
          }
        }
        for (c6=32*c3;c6<=32*c3+31;c6++) {
          {hz[-c4+c6-1][ny-1]=hz[-c4+c6-1][ny-1]-((double)(7))/10*(ey[1+-c4+c6-1][ny-1]+ex[-c4+c6-1][1+ny-1]-ex[-c4+c6-1][ny-1]-ey[-c4+c6-1][ny-1]);} ;
        }
      }
      for (c4=max(max(max(max(32*c2,0),32*c1-32*c2),32*c3-nx+1),32*c2-ny+32);c4<=min(min(min(min(tmax-1,32*c3-1),32*c1-32*c2+31),32*c2+30),32*c3-nx+31);c4++) {
        for (c6=32*c3;c6<=c4+nx-1;c6++) {
          {ey[-c4+c6][0]=ey[-c4+c6][0]-((double)(1))/2*(hz[-c4+c6][0]-hz[-c4+c6-1][0]);} ;
        }
        for (c5=c4+1;c5<=32*c2+31;c5++) {
          for (c6=32*c3;c6<=c4+nx-1;c6++) {
            {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
            {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
            {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
          }
          {hz[nx-1][-c4+c5-1]=hz[nx-1][-c4+c5-1]-((double)(7))/10*(ey[1+nx-1][-c4+c5-1]+ex[nx-1][1+-c4+c5-1]-ex[nx-1][-c4+c5-1]-ey[nx-1][-c4+c5-1]);} ;
        }
      }
      if (ny >= 2) {
        for (c4=max(max(max(32*c2,32*c3-nx+32),0),32*c1-32*c2);c4<=min(min(min(32*c3-1,tmax-1),32*c1-32*c2+31),32*c2-ny+31);c4++) {
          for (c6=32*c3;c6<=32*c3+31;c6++) {
            {ey[-c4+c6][0]=ey[-c4+c6][0]-((double)(1))/2*(hz[-c4+c6][0]-hz[-c4+c6-1][0]);} ;
          }
          for (c5=c4+1;c5<=c4+ny-1;c5++) {
            for (c6=32*c3;c6<=32*c3+31;c6++) {
              {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
              {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
              {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
            }
          }
          for (c6=32*c3;c6<=32*c3+31;c6++) {
            {hz[-c4+c6-1][ny-1]=hz[-c4+c6-1][ny-1]-((double)(7))/10*(ey[1+-c4+c6-1][ny-1]+ex[-c4+c6-1][1+ny-1]-ex[-c4+c6-1][ny-1]-ey[-c4+c6-1][ny-1]);} ;
          }
        }
      }
      for (c4=max(max(max(0,32*c1-32*c2),32*c3-nx+32),32*c2-ny+32);c4<=min(min(min(tmax-1,32*c3-1),32*c1-32*c2+31),32*c2-1);c4++) {

	/*@ begin Loop(
	  transform UnrollJam(ufactor=4)
	  for (c5=32*c2;c5<=32*c2+31;c5++) {
	  transform UnrollJam(ufactor=4)
          for (c6=32*c3;c6<=32*c3+31;c6++) {
	    ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);
            ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);
            hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);
          }
        }
	) @*/ {
   for (c5=32*c2; c5<=32*c2+28; c5=c5+4) {
     for (c6=32*c3; c6<=32*c3+28; c6=c6+4) {
       ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);
       ey[-c4+c6][-c4+c5+1]=ey[-c4+c6][-c4+c5+1]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+1]-hz[-c4+c6-1][-c4+c5+1]);
       ey[-c4+c6][-c4+c5+2]=ey[-c4+c6][-c4+c5+2]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+2]-hz[-c4+c6-1][-c4+c5+2]);
       ey[-c4+c6][-c4+c5+3]=ey[-c4+c6][-c4+c5+3]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+3]-hz[-c4+c6-1][-c4+c5+3]);
       ey[-c4+c6+1][-c4+c5]=ey[-c4+c6+1][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+1][-c4+c5]-hz[-c4+c6][-c4+c5]);
       ey[-c4+c6+1][-c4+c5+1]=ey[-c4+c6+1][-c4+c5+1]-0.5*((double)(1))*(hz[-c4+c6+1][-c4+c5+1]-hz[-c4+c6][-c4+c5+1]);
       ey[-c4+c6+1][-c4+c5+2]=ey[-c4+c6+1][-c4+c5+2]-0.5*((double)(1))*(hz[-c4+c6+1][-c4+c5+2]-hz[-c4+c6][-c4+c5+2]);
       ey[-c4+c6+1][-c4+c5+3]=ey[-c4+c6+1][-c4+c5+3]-0.5*((double)(1))*(hz[-c4+c6+1][-c4+c5+3]-hz[-c4+c6][-c4+c5+3]);
       ey[-c4+c6+2][-c4+c5]=ey[-c4+c6+2][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+2][-c4+c5]-hz[-c4+c6+1][-c4+c5]);
       ey[-c4+c6+2][-c4+c5+1]=ey[-c4+c6+2][-c4+c5+1]-0.5*((double)(1))*(hz[-c4+c6+2][-c4+c5+1]-hz[-c4+c6+1][-c4+c5+1]);
       ey[-c4+c6+2][-c4+c5+2]=ey[-c4+c6+2][-c4+c5+2]-0.5*((double)(1))*(hz[-c4+c6+2][-c4+c5+2]-hz[-c4+c6+1][-c4+c5+2]);
       ey[-c4+c6+2][-c4+c5+3]=ey[-c4+c6+2][-c4+c5+3]-0.5*((double)(1))*(hz[-c4+c6+2][-c4+c5+3]-hz[-c4+c6+1][-c4+c5+3]);
       ey[-c4+c6+3][-c4+c5]=ey[-c4+c6+3][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+3][-c4+c5]-hz[-c4+c6+2][-c4+c5]);
       ey[-c4+c6+3][-c4+c5+1]=ey[-c4+c6+3][-c4+c5+1]-0.5*((double)(1))*(hz[-c4+c6+3][-c4+c5+1]-hz[-c4+c6+2][-c4+c5+1]);
       ey[-c4+c6+3][-c4+c5+2]=ey[-c4+c6+3][-c4+c5+2]-0.5*((double)(1))*(hz[-c4+c6+3][-c4+c5+2]-hz[-c4+c6+2][-c4+c5+2]);
       ey[-c4+c6+3][-c4+c5+3]=ey[-c4+c6+3][-c4+c5+3]-0.5*((double)(1))*(hz[-c4+c6+3][-c4+c5+3]-hz[-c4+c6+2][-c4+c5+3]);
       ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);
       ex[-c4+c6][-c4+c5+1]=ex[-c4+c6][-c4+c5+1]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+1]-hz[-c4+c6][-c4+c5]);
       ex[-c4+c6][-c4+c5+2]=ex[-c4+c6][-c4+c5+2]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+2]-hz[-c4+c6][-c4+c5+1]);
       ex[-c4+c6][-c4+c5+3]=ex[-c4+c6][-c4+c5+3]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+3]-hz[-c4+c6][-c4+c5+2]);
       ex[-c4+c6+1][-c4+c5]=ex[-c4+c6+1][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+1][-c4+c5]-hz[-c4+c6+1][-c4+c5-1]);
       ex[-c4+c6+1][-c4+c5+1]=ex[-c4+c6+1][-c4+c5+1]-0.5*((double)(1))*(hz[-c4+c6+1][-c4+c5+1]-hz[-c4+c6+1][-c4+c5]);
       ex[-c4+c6+1][-c4+c5+2]=ex[-c4+c6+1][-c4+c5+2]-0.5*((double)(1))*(hz[-c4+c6+1][-c4+c5+2]-hz[-c4+c6+1][-c4+c5+1]);
       ex[-c4+c6+1][-c4+c5+3]=ex[-c4+c6+1][-c4+c5+3]-0.5*((double)(1))*(hz[-c4+c6+1][-c4+c5+3]-hz[-c4+c6+1][-c4+c5+2]);
       ex[-c4+c6+2][-c4+c5]=ex[-c4+c6+2][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+2][-c4+c5]-hz[-c4+c6+2][-c4+c5-1]);
       ex[-c4+c6+2][-c4+c5+1]=ex[-c4+c6+2][-c4+c5+1]-0.5*((double)(1))*(hz[-c4+c6+2][-c4+c5+1]-hz[-c4+c6+2][-c4+c5]);
       ex[-c4+c6+2][-c4+c5+2]=ex[-c4+c6+2][-c4+c5+2]-0.5*((double)(1))*(hz[-c4+c6+2][-c4+c5+2]-hz[-c4+c6+2][-c4+c5+1]);
       ex[-c4+c6+2][-c4+c5+3]=ex[-c4+c6+2][-c4+c5+3]-0.5*((double)(1))*(hz[-c4+c6+2][-c4+c5+3]-hz[-c4+c6+2][-c4+c5+2]);
       ex[-c4+c6+3][-c4+c5]=ex[-c4+c6+3][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+3][-c4+c5]-hz[-c4+c6+3][-c4+c5-1]);
       ex[-c4+c6+3][-c4+c5+1]=ex[-c4+c6+3][-c4+c5+1]-0.5*((double)(1))*(hz[-c4+c6+3][-c4+c5+1]-hz[-c4+c6+3][-c4+c5]);
       ex[-c4+c6+3][-c4+c5+2]=ex[-c4+c6+3][-c4+c5+2]-0.5*((double)(1))*(hz[-c4+c6+3][-c4+c5+2]-hz[-c4+c6+3][-c4+c5+1]);
       ex[-c4+c6+3][-c4+c5+3]=ex[-c4+c6+3][-c4+c5+3]-0.5*((double)(1))*(hz[-c4+c6+3][-c4+c5+3]-hz[-c4+c6+3][-c4+c5+2]);
       hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-0.1*((double)(7))*(ey[-c4+c6][-c4+c5-1]+ex[-c4+c6-1][-c4+c5]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);
       hz[-c4+c6-1][-c4+c5]=hz[-c4+c6-1][-c4+c5]-0.1*((double)(7))*(ey[-c4+c6][-c4+c5]+ex[-c4+c6-1][-c4+c5+1]-ex[-c4+c6-1][-c4+c5]-ey[-c4+c6-1][-c4+c5]);
       hz[-c4+c6-1][-c4+c5+1]=hz[-c4+c6-1][-c4+c5+1]-0.1*((double)(7))*(ey[-c4+c6][-c4+c5+1]+ex[-c4+c6-1][-c4+c5+2]-ex[-c4+c6-1][-c4+c5+1]-ey[-c4+c6-1][-c4+c5+1]);
       hz[-c4+c6-1][-c4+c5+2]=hz[-c4+c6-1][-c4+c5+2]-0.1*((double)(7))*(ey[-c4+c6][-c4+c5+2]+ex[-c4+c6-1][-c4+c5+3]-ex[-c4+c6-1][-c4+c5+2]-ey[-c4+c6-1][-c4+c5+2]);
       hz[-c4+c6][-c4+c5-1]=hz[-c4+c6][-c4+c5-1]-0.1*((double)(7))*(ey[-c4+c6+1][-c4+c5-1]+ex[-c4+c6][-c4+c5]-ex[-c4+c6][-c4+c5-1]-ey[-c4+c6][-c4+c5-1]);
       hz[-c4+c6][-c4+c5]=hz[-c4+c6][-c4+c5]-0.1*((double)(7))*(ey[-c4+c6+1][-c4+c5]+ex[-c4+c6][-c4+c5+1]-ex[-c4+c6][-c4+c5]-ey[-c4+c6][-c4+c5]);
       hz[-c4+c6][-c4+c5+1]=hz[-c4+c6][-c4+c5+1]-0.1*((double)(7))*(ey[-c4+c6+1][-c4+c5+1]+ex[-c4+c6][-c4+c5+2]-ex[-c4+c6][-c4+c5+1]-ey[-c4+c6][-c4+c5+1]);
       hz[-c4+c6][-c4+c5+2]=hz[-c4+c6][-c4+c5+2]-0.1*((double)(7))*(ey[-c4+c6+1][-c4+c5+2]+ex[-c4+c6][-c4+c5+3]-ex[-c4+c6][-c4+c5+2]-ey[-c4+c6][-c4+c5+2]);
       hz[-c4+c6+1][-c4+c5-1]=hz[-c4+c6+1][-c4+c5-1]-0.1*((double)(7))*(ey[-c4+c6+2][-c4+c5-1]+ex[-c4+c6+1][-c4+c5]-ex[-c4+c6+1][-c4+c5-1]-ey[-c4+c6+1][-c4+c5-1]);
       hz[-c4+c6+1][-c4+c5]=hz[-c4+c6+1][-c4+c5]-0.1*((double)(7))*(ey[-c4+c6+2][-c4+c5]+ex[-c4+c6+1][-c4+c5+1]-ex[-c4+c6+1][-c4+c5]-ey[-c4+c6+1][-c4+c5]);
       hz[-c4+c6+1][-c4+c5+1]=hz[-c4+c6+1][-c4+c5+1]-0.1*((double)(7))*(ey[-c4+c6+2][-c4+c5+1]+ex[-c4+c6+1][-c4+c5+2]-ex[-c4+c6+1][-c4+c5+1]-ey[-c4+c6+1][-c4+c5+1]);
       hz[-c4+c6+1][-c4+c5+2]=hz[-c4+c6+1][-c4+c5+2]-0.1*((double)(7))*(ey[-c4+c6+2][-c4+c5+2]+ex[-c4+c6+1][-c4+c5+3]-ex[-c4+c6+1][-c4+c5+2]-ey[-c4+c6+1][-c4+c5+2]);
       hz[-c4+c6+2][-c4+c5-1]=hz[-c4+c6+2][-c4+c5-1]-0.1*((double)(7))*(ey[-c4+c6+3][-c4+c5-1]+ex[-c4+c6+2][-c4+c5]-ex[-c4+c6+2][-c4+c5-1]-ey[-c4+c6+2][-c4+c5-1]);
       hz[-c4+c6+2][-c4+c5]=hz[-c4+c6+2][-c4+c5]-0.1*((double)(7))*(ey[-c4+c6+3][-c4+c5]+ex[-c4+c6+2][-c4+c5+1]-ex[-c4+c6+2][-c4+c5]-ey[-c4+c6+2][-c4+c5]);
       hz[-c4+c6+2][-c4+c5+1]=hz[-c4+c6+2][-c4+c5+1]-0.1*((double)(7))*(ey[-c4+c6+3][-c4+c5+1]+ex[-c4+c6+2][-c4+c5+2]-ex[-c4+c6+2][-c4+c5+1]-ey[-c4+c6+2][-c4+c5+1]);
       hz[-c4+c6+2][-c4+c5+2]=hz[-c4+c6+2][-c4+c5+2]-0.1*((double)(7))*(ey[-c4+c6+3][-c4+c5+2]+ex[-c4+c6+2][-c4+c5+3]-ex[-c4+c6+2][-c4+c5+2]-ey[-c4+c6+2][-c4+c5+2]);
     }
     for (; c6<=32*c3+31; c6=c6+1) {
       ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);
       ey[-c4+c6][-c4+c5+1]=ey[-c4+c6][-c4+c5+1]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+1]-hz[-c4+c6-1][-c4+c5+1]);
       ey[-c4+c6][-c4+c5+2]=ey[-c4+c6][-c4+c5+2]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+2]-hz[-c4+c6-1][-c4+c5+2]);
       ey[-c4+c6][-c4+c5+3]=ey[-c4+c6][-c4+c5+3]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+3]-hz[-c4+c6-1][-c4+c5+3]);
       ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);
       ex[-c4+c6][-c4+c5+1]=ex[-c4+c6][-c4+c5+1]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+1]-hz[-c4+c6][-c4+c5]);
       ex[-c4+c6][-c4+c5+2]=ex[-c4+c6][-c4+c5+2]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+2]-hz[-c4+c6][-c4+c5+1]);
       ex[-c4+c6][-c4+c5+3]=ex[-c4+c6][-c4+c5+3]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5+3]-hz[-c4+c6][-c4+c5+2]);
       hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-0.1*((double)(7))*(ey[-c4+c6][-c4+c5-1]+ex[-c4+c6-1][-c4+c5]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);
       hz[-c4+c6-1][-c4+c5]=hz[-c4+c6-1][-c4+c5]-0.1*((double)(7))*(ey[-c4+c6][-c4+c5]+ex[-c4+c6-1][-c4+c5+1]-ex[-c4+c6-1][-c4+c5]-ey[-c4+c6-1][-c4+c5]);
       hz[-c4+c6-1][-c4+c5+1]=hz[-c4+c6-1][-c4+c5+1]-0.1*((double)(7))*(ey[-c4+c6][-c4+c5+1]+ex[-c4+c6-1][-c4+c5+2]-ex[-c4+c6-1][-c4+c5+1]-ey[-c4+c6-1][-c4+c5+1]);
       hz[-c4+c6-1][-c4+c5+2]=hz[-c4+c6-1][-c4+c5+2]-0.1*((double)(7))*(ey[-c4+c6][-c4+c5+2]+ex[-c4+c6-1][-c4+c5+3]-ex[-c4+c6-1][-c4+c5+2]-ey[-c4+c6-1][-c4+c5+2]);
     }
   }
   for (; c5<=32*c2+31; c5=c5+1) {
     {
       for (c6=32*c3; c6<=32*c3+28; c6=c6+4) {
         ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);
         ey[-c4+c6+1][-c4+c5]=ey[-c4+c6+1][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+1][-c4+c5]-hz[-c4+c6][-c4+c5]);
         ey[-c4+c6+2][-c4+c5]=ey[-c4+c6+2][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+2][-c4+c5]-hz[-c4+c6+1][-c4+c5]);
         ey[-c4+c6+3][-c4+c5]=ey[-c4+c6+3][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+3][-c4+c5]-hz[-c4+c6+2][-c4+c5]);
         ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);
         ex[-c4+c6+1][-c4+c5]=ex[-c4+c6+1][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+1][-c4+c5]-hz[-c4+c6+1][-c4+c5-1]);
         ex[-c4+c6+2][-c4+c5]=ex[-c4+c6+2][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+2][-c4+c5]-hz[-c4+c6+2][-c4+c5-1]);
         ex[-c4+c6+3][-c4+c5]=ex[-c4+c6+3][-c4+c5]-0.5*((double)(1))*(hz[-c4+c6+3][-c4+c5]-hz[-c4+c6+3][-c4+c5-1]);
         hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-0.1*((double)(7))*(ey[-c4+c6][-c4+c5-1]+ex[-c4+c6-1][-c4+c5]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);
         hz[-c4+c6][-c4+c5-1]=hz[-c4+c6][-c4+c5-1]-0.1*((double)(7))*(ey[-c4+c6+1][-c4+c5-1]+ex[-c4+c6][-c4+c5]-ex[-c4+c6][-c4+c5-1]-ey[-c4+c6][-c4+c5-1]);
         hz[-c4+c6+1][-c4+c5-1]=hz[-c4+c6+1][-c4+c5-1]-0.1*((double)(7))*(ey[-c4+c6+2][-c4+c5-1]+ex[-c4+c6+1][-c4+c5]-ex[-c4+c6+1][-c4+c5-1]-ey[-c4+c6+1][-c4+c5-1]);
         hz[-c4+c6+2][-c4+c5-1]=hz[-c4+c6+2][-c4+c5-1]-0.1*((double)(7))*(ey[-c4+c6+3][-c4+c5-1]+ex[-c4+c6+2][-c4+c5]-ex[-c4+c6+2][-c4+c5-1]-ey[-c4+c6+2][-c4+c5-1]);
       }
       for (; c6<=32*c3+31; c6=c6+1) {
         ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);
         ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);
         hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);
       }
     }
   }
 }
/*@ end @*/
      }
      for (c4=max(max(max(max(32*c2,32*c3-nx+32),0),32*c1-32*c2),32*c2-ny+32);c4<=min(min(min(tmax-1,32*c3-1),32*c1-32*c2+31),32*c2+30);c4++) {
        for (c6=32*c3;c6<=32*c3+31;c6++) {
          {ey[-c4+c6][0]=ey[-c4+c6][0]-((double)(1))/2*(hz[-c4+c6][0]-hz[-c4+c6-1][0]);} ;
        }
        for (c5=c4+1;c5<=32*c2+31;c5++) {
          for (c6=32*c3;c6<=32*c3+31;c6++) {
            {ey[-c4+c6][-c4+c5]=ey[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6-1][-c4+c5]);} ;
            {ex[-c4+c6][-c4+c5]=ex[-c4+c6][-c4+c5]-((double)(1))/2*(hz[-c4+c6][-c4+c5]-hz[-c4+c6][-c4+c5-1]);} ;
            {hz[-c4+c6-1][-c4+c5-1]=hz[-c4+c6-1][-c4+c5-1]-((double)(7))/10*(ey[1+-c4+c6-1][-c4+c5-1]+ex[-c4+c6-1][1+-c4+c5-1]-ex[-c4+c6-1][-c4+c5-1]-ey[-c4+c6-1][-c4+c5-1]);} ;
          }
        }
      }
      if (ny == 1) {
        for (c4=max(max(max(0,32*c3-nx+1),32*c1-32*c2),32*c2);c4<=min(min(min(32*c3-1,tmax-1),32*c1-32*c2+31),32*c2+30);c4++) {
          for (c6=32*c3;c6<=min(c4+nx-1,32*c3+31);c6++) {
            {ey[-c4+c6][0]=ey[-c4+c6][0]-((double)(1))/2*(hz[-c4+c6][0]-hz[-c4+c6-1][0]);} ;
          }
          for (c6=32*c3;c6<=min(32*c3+31,c4+nx);c6++) {
            {hz[-c4+c6-1][0]=hz[-c4+c6-1][0]-((double)(7))/10*(ey[1+-c4+c6-1][0]+ex[-c4+c6-1][1+0]-ex[-c4+c6-1][0]-ey[-c4+c6-1][0]);} ;
          }
        }
      }
      if ((-c1 == -c2-c3) && (c1 <= min(floord(64*c3-1,32),floord(32*c3+tmax-32,32))) && (nx >= 2) && (ny == 1)) {
        {ey[0][0]=32*c1-32*c3+31;} ;
        for (c6=32*c1-32*c3+32;c6<=min(32*c1-32*c3+nx+30,32*c3+31);c6++) {
          {ey[-32*c1+32*c3+c6-31][0]=ey[-32*c1+32*c3+c6-31][0]-((double)(1))/2*(hz[-32*c1+32*c3+c6-31][0]-hz[-32*c1+32*c3+c6-31 -1][0]);} ;
        }
      }
      if ((-c1 == -c2-c3) && (c1 <= min(floord(64*c3-1,32),floord(32*c3+tmax-32,32))) && (nx >= 2) && (ny >= 2)) {
        {ey[0][0]=32*c1-32*c3+31;} ;
        for (c6=32*c1-32*c3+32;c6<=min(32*c1-32*c3+nx+30,32*c3+31);c6++) {
          {ey[-32*c1+32*c3+c6-31][0]=ey[-32*c1+32*c3+c6-31][0]-((double)(1))/2*(hz[-32*c1+32*c3+c6-31][0]-hz[-32*c1+32*c3+c6-31 -1][0]);} ;
        }
      }
      if ((-c1 == -c2-c3) && (c1 >= ceild(64*c2-31,32)) && (c1 <= min(floord(32*c2+tmax-32,32),floord(64*c2-1,32))) && (nx == 1) && (ny >= 2)) {
        {ey[0][0]=32*c1-32*c2+31;} ;
        for (c5=32*c1-32*c2+32;c5<=min(32*c2+31,32*c1-32*c2+ny+30);c5++) {
          {ey[0][-32*c1+32*c2+c5-31]=32*c1-32*c2+31;} ;
          {ex[0][-32*c1+32*c2+c5-31]=ex[0][-32*c1+32*c2+c5-31]-((double)(1))/2*(hz[0][-32*c1+32*c2+c5-31]-hz[0][-32*c1+32*c2+c5-31 -1]);} ;
        }
      }
      if ((-c1 == -c2-c3) && (c1 <= min(floord(32*c2+tmax-32,32),2*c2-1)) && (nx == 1)) {
        for (c5=32*c2;c5<=min(32*c2+31,32*c1-32*c2+ny+30);c5++) {
          {ey[0][-32*c1+32*c2+c5-31]=32*c1-32*c2+31;} ;
          {ex[0][-32*c1+32*c2+c5-31]=ex[0][-32*c1+32*c2+c5-31]-((double)(1))/2*(hz[0][-32*c1+32*c2+c5-31]-hz[0][-32*c1+32*c2+c5-31 -1]);} ;
        }
      }
      if ((-c1 == -c2-c3) && (c1 <= min(floord(32*c3+tmax-32,32),2*c3)) && (nx == 1) && (ny == 1)) {
        {ey[0][0]=32*c1-32*c3+31;} ;
      }
      if ((-c1 == -c2-c3) && (c1 <= min(floord(32*c2+tmax-32,32),floord(64*c2-1,32))) && (nx == 1) && (ny == 1)) {
        {ey[0][0]=32*c1-32*c2+31;} ;
      }
      if ((-c1 == -c2-c3) && (c1 <= min(floord(32*c3+tmax-32,32),2*c3)) && (nx == 1) && (ny >= 2)) {
        {ey[0][0]=32*c1-32*c3+31;} ;
      }
      if ((-c1 == -c2-c3) && (c1 >= ceild(64*c2-31,32)) && (c1 <= min(floord(32*c2+tmax-32,32),floord(64*c2-1,32))) && (nx >= 2) && (ny >= 2)) {
        {ey[0][0]=32*c1-32*c2+31;} ;
        for (c5=32*c1-32*c2+32;c5<=min(32*c2+31,32*c1-32*c2+ny+30);c5++) {
          {ey[0][-32*c1+32*c2+c5-31]=32*c1-32*c2+31;} ;
          {ex[0][-32*c1+32*c2+c5-31]=ex[0][-32*c1+32*c2+c5-31]-((double)(1))/2*(hz[0][-32*c1+32*c2+c5-31]-hz[0][-32*c1+32*c2+c5-31 -1]);} ;
        }
      }
      if ((-c1 == -c2-c3) && (c1 <= min(floord(32*c2+tmax-32,32),2*c2-1)) && (nx >= 2)) {
        for (c5=32*c2;c5<=min(32*c2+31,32*c1-32*c2+ny+30);c5++) {
          {ey[0][-32*c1+32*c2+c5-31]=32*c1-32*c2+31;} ;
          {ex[0][-32*c1+32*c2+c5-31]=ex[0][-32*c1+32*c2+c5-31]-((double)(1))/2*(hz[0][-32*c1+32*c2+c5-31]-hz[0][-32*c1+32*c2+c5-31 -1]);} ;
        }
      }
      if ((-c1 == -c2-c3) && (c1 <= min(floord(32*c2+tmax-32,32),2*c2)) && (nx >= 2) && (ny == 1)) {
        {ey[0][0]=32*c1-32*c2+31;} ;
      }
      if ((-c1 == -2*c2) && (-c1 == -2*c3) && (c1 <= floord(tmax-32,16)) && (nx >= 2) && (ny >= 2)) {
        if (c1%2 == 0) {
          {ey[0][0]=16*c1+31;} ;
        }
      }
      if ((c1 >= 2*c2) && (c2 <= min(floord(tmax-32,32),c3-1)) && (ny == 1)) {
        for (c6=32*c3;c6<=min(32*c2+nx+30,32*c3+31);c6++) {
          {ey[-32*c2+c6-31][0]=ey[-32*c2+c6-31][0]-((double)(1))/2*(hz[-32*c2+c6-31][0]-hz[-32*c2+c6-31 -1][0]);} ;
        }
      }
      if ((c1 >= 2*c2) && (c2 <= min(floord(tmax-32,32),c3-1)) && (ny >= 2)) {
        for (c6=32*c3;c6<=min(32*c2+nx+30,32*c3+31);c6++) {
          {ey[-32*c2+c6-31][0]=ey[-32*c2+c6-31][0]-((double)(1))/2*(hz[-32*c2+c6-31][0]-hz[-32*c2+c6-31 -1][0]);} ;
        }
      }
    }
  }
}



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
                                    
