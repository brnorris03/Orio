
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
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))





 int c1, c2, c3, c4, c5, c6, c7, c8, c9;
 register int lbv, ubv;

for (c1=0;c1<=floord(tmax-1,256);c1++) {
  for (c2=max(0,ceild(256*c1-255,256));c2<=min(floord(256*c1+ny+255,256),floord(tmax+ny-1,256));c2++) {
    for (c3=max(max(max(max(max(max(max(max(max(max(ceild(256*c2-ny-254,256),ceild(256*c1-255*ny-64515,65280)),ceild(256*c1-255,256)),ceild(65280*c1-65024*c2-254*nx-ny-64515,256)),ceild(65536*c1-65280*c2-254*nx-64771,256)),0),ceild(512*c1-256*c2-509,256)),ceild(256*c1-ny-253,256)),ceild(256*c1-65280*c2-129795,65280)),ceild(256*c1-65280*c2-254*nx-64771,256)),ceild(256*c1-65024*c2-254*nx-ny-64515,256));c3<=min(min(min(floord(256*c1+nx+255,256),c1+256*c2+nx+254),floord(tmax+nx-1,256)),floord(256*c2+nx+254,256));c3++) {
      for (c4=max(max(max(max(max(max(max(max(8*c1,0),ceild(256*c1-57344*c2-256*c3-223*nx-56928,32)),ceild(256*c2-ny-31,32)),ceild(256*c1-57088*c2-256*c3-223*nx-ny-56672,32)),8*c1-1792*c3-7*ny-1771),ceild(256*c3-nx-31,32)),8*c1-1792*c2-1792*c3-3563),-256*c2+8*c3-nx-254);c4<=min(min(min(min(min(min(min(min(min(min(min(min(min(min(min(floord(256*c2+255,32),floord(tmax-1,32)),floord(256*c2+256*c3+509,64)),floord(256*c3+ny+253,32)),floord(7936*c2+7936*c3+15779,32)),floord(7936*c3+31*ny+7843,32)),floord(256*c3+255,32)),8*c1+7),floord(-256*c1+65024*c2+256*c3+254*nx+ny+64515,8128)),floord(-256*c1+65280*c2+256*c3+254*nx+64771,8160)),floord(256*c1+57600*c2-256*c3+225*nx+57150,32)),floord(256*c1+57600*c2-256*c3+225*nx+57150,7200)),floord(7680*c2+256*c3+30*nx+ny+7843,32)),floord(7936*c2+256*c3+30*nx+8099,32)),floord(7680*c2+256*c3+30*nx+ny+7843,992)),floord(7936*c2+256*c3+30*nx+8099,1024));c4++) {
        for (c5=max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(ceild(32*c4-31,32),0),ceild(256*c1-256*c3-254*nx-ny-7619,8128)),ceild(256*c1-256*c3-254*nx-7651,8160)),ceild(-c1+c3-nx-30,32)),8*c2),ceild(256*c1-256*c3+8128*c4-254*nx-ny-7619,8128)),ceild(256*c1-256*c3+8160*c4-254*nx-7651,8160)),ceild(-256*c1+256*c3+7200*c4-225*nx-6750,7200)),ceild(-256*c1+256*c3+32*c4-225*nx-6750,7200)),ceild(256*c3-nx-30,32)),ceild(8*c3-c4-nx-30,32)),ceild(-256*c3+32*c4-30*nx-1155,992)),ceild(-256*c3+992*c4-30*nx-ny-1123,960)),ceild(256*c1-256*c3-32*c4-223*nx-ny-6720,7136)),ceild(-256*c3+64*c4-285,32)),ceild(-256*c3+32*c4-30*nx-ny-1123,960)),ceild(-7936*c3+32*c4-8835,992)),ceild(256*c1-256*c3-32*c4-223*nx-6752,7168)),ceild(8*c1-1792*c3-c4-1995,224)),ceild(-256*c3+1024*c4-30*nx-1155,992));c5<=min(min(min(floord(32*c4+ny+31,32),floord(tmax+ny-1,32)),8*c2+7),floord(256*c3+ny+254,32));c5++) {
          for (c6=max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(max(ceild(1024*c4-992*c5-30*nx-931,32),ceild(32*c4-960*c5-30*nx-ny-899,32)),ceild(992*c4-960*c5-30*nx-ny-899,32)),ceild(32*c4-992*c5-30*nx-931,32)),-8*c1+8*c3+c4-224*c5-7*nx-210),-8*c1+8*c3+225*c4-224*c5-7*nx-210),ceild(32*c4-31,32)),ceild(-8*c1+8*c3+c4-7*ny-210,225)),8*c3),ceild(32*c4-ny-29,32)),ceild(32*c4-992*c5-1891,992)),ceild(32*c5-ny-30,32)),ceild(8*c1-8*c3-c4-7*ny-210,223)),ceild(64*c4-32*c5-61,32)),ceild(-8*c1+8*c3+c4-224*c5-434,225)),ceild(8*c1-8*c3-c4-224*c5-434,223)),0),ceild(32*c4-31*ny-899,992));c6<=min(min(min(min(min(min(8*c3+7,floord(32*c5+nx+30,32)),c4+32*c5+nx+30),-8*c1+8*c3+c4+224*c5+7*nx+210),-8*c1+8*c3-223*c4+224*c5+7*nx+210),floord(tmax+nx-1,32)),floord(32*c4+nx+31,32));c6++) {
            if ((c4 <= floord(32*c6-nx,32)) && (c5 <= floord(32*c6-nx+ny,32)) && (c6 >= ceild(nx,32))) {
              for (c8=max(32*c6-nx+1,32*c5);c8<=min(32*c6-nx+ny,32*c5+31);c8++) {
                {hz[nx-1][-32*c6+c8+nx-1]=hz[nx-1][-32*c6+c8+nx-1]-((double)(7))/10*(ey[1+nx-1][-32*c6+c8+nx-1]+ex[nx-1][1+-32*c6+c8+nx-1]-ex[nx-1][-32*c6+c8+nx-1]-ey[nx-1][-32*c6+c8+nx-1]);} ;
              }
            }
            if ((c4 <= floord(32*c5-ny,32)) && (c5 >= max(ceild(ny,32),ceild(32*c6-nx+ny+1,32)))) {
              for (c9=max(32*c5-ny+1,32*c6);c9<=min(32*c5+nx-ny,32*c6+31);c9++) {
                {hz[-32*c5+c9+ny-1][ny-1]=hz[-32*c5+c9+ny-1][ny-1]-((double)(7))/10*(ey[1+-32*c5+c9+ny-1][ny-1]+ex[-32*c5+c9+ny-1][1+ny-1]-ex[-32*c5+c9+ny-1][ny-1]-ey[-32*c5+c9+ny-1][ny-1]);} ;
              }
            }
            if ((c1 == c3) && (c4 == c6) && (nx >= 2)) {
              for (c7=max(max(0,32*c6),32*c5-ny+1);c7<=min(min(min(32*c5-1,32*c6-nx+31),tmax-1),32*c5-ny+31);c7++) {
                for (c8=32*c5;c8<=c7+ny-1;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  for (c9=c7+1;c9<=c7+nx-1;c9++) {
                    {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                    {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                    {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                  }
                  {hz[nx-1][-c7+c8-1]=hz[nx-1][-c7+c8-1]-((double)(7))/10*(ey[1+nx-1][-c7+c8-1]+ex[nx-1][1+-c7+c8-1]-ex[nx-1][-c7+c8-1]-ey[nx-1][-c7+c8-1]);} ;
                }
                for (c9=c7+1;c9<=c7+nx;c9++) {
                  {hz[-c7+c9-1][ny-1]=hz[-c7+c9-1][ny-1]-((double)(7))/10*(ey[1+-c7+c9-1][ny-1]+ex[-c7+c9-1][1+ny-1]-ex[-c7+c9-1][ny-1]-ey[-c7+c9-1][ny-1]);} ;
                }
              }
            }
            if ((c1 == c3) && (c4 == c6) && (nx >= 2) && (ny >= 2)) {
              for (c7=max(max(32*c5,0),32*c6);c7<=min(min(32*c6-nx+31,tmax-1),32*c5-ny+31);c7++) {
                {ey[0][0]=c7;} ;
                for (c9=c7+1;c9<=c7+nx-1;c9++) {
                  {ey[-c7+c9][0]=ey[-c7+c9][0]-((double)(1))/2*(hz[-c7+c9][0]-hz[-c7+c9-1][0]);} ;
                }
                for (c8=c7+1;c8<=c7+ny-1;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  for (c9=c7+1;c9<=c7+nx-1;c9++) {
                    {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                    {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                    {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                  }
                  {hz[nx-1][-c7+c8-1]=hz[nx-1][-c7+c8-1]-((double)(7))/10*(ey[1+nx-1][-c7+c8-1]+ex[nx-1][1+-c7+c8-1]-ex[nx-1][-c7+c8-1]-ey[nx-1][-c7+c8-1]);} ;
                }
                for (c9=c7+1;c9<=c7+nx;c9++) {
                  {hz[-c7+c9-1][ny-1]=hz[-c7+c9-1][ny-1]-((double)(7))/10*(ey[1+-c7+c9-1][ny-1]+ex[-c7+c9-1][1+ny-1]-ex[-c7+c9-1][ny-1]-ey[-c7+c9-1][ny-1]);} ;
                }
              }
            }
            if ((c1 == c3) && (c4 == c6) && (nx >= 2)) {
              for (c7=max(max(0,32*c6),32*c5-ny+32);c7<=min(min(tmax-1,32*c5-1),32*c6-nx+31);c7++) {
                for (c8=32*c5;c8<=32*c5+31;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  for (c9=c7+1;c9<=c7+nx-1;c9++) {
                    {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                    {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                    {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                  }
                  {hz[nx-1][-c7+c8-1]=hz[nx-1][-c7+c8-1]-((double)(7))/10*(ey[1+nx-1][-c7+c8-1]+ex[nx-1][1+-c7+c8-1]-ex[nx-1][-c7+c8-1]-ey[nx-1][-c7+c8-1]);} ;
                }
              }
            }
            if ((c1 == c3) && (c4 == c6)) {
              for (c7=max(max(max(32*c6-nx+32,0),32*c6),32*c5-ny+1);c7<=min(min(min(32*c5-1,tmax-1),32*c5-ny+31),32*c6+30);c7++) {
                for (c8=32*c5;c8<=c7+ny-1;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  for (c9=c7+1;c9<=32*c6+31;c9++) {
                    {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                    {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                    {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                  }
                }
                for (c9=c7+1;c9<=32*c6+31;c9++) {
                  {hz[-c7+c9-1][ny-1]=hz[-c7+c9-1][ny-1]-((double)(7))/10*(ey[1+-c7+c9-1][ny-1]+ex[-c7+c9-1][1+ny-1]-ex[-c7+c9-1][ny-1]-ey[-c7+c9-1][ny-1]);} ;
                }
              }
            }
            if ((c1 == c3) && (c4 == c6) && (nx >= 2)) {
              for (c7=max(max(max(32*c5,0),32*c6),32*c5-ny+32);c7<=min(min(tmax-1,32*c5+30),32*c6-nx+31);c7++) {
                {ey[0][0]=c7;} ;
                for (c9=c7+1;c9<=c7+nx-1;c9++) {
                  {ey[-c7+c9][0]=ey[-c7+c9][0]-((double)(1))/2*(hz[-c7+c9][0]-hz[-c7+c9-1][0]);} ;
                }
                for (c8=c7+1;c8<=32*c5+31;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  for (c9=c7+1;c9<=c7+nx-1;c9++) {
                    {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                    {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                    {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                  }
                  {hz[nx-1][-c7+c8-1]=hz[nx-1][-c7+c8-1]-((double)(7))/10*(ey[1+nx-1][-c7+c8-1]+ex[nx-1][1+-c7+c8-1]-ex[nx-1][-c7+c8-1]-ey[nx-1][-c7+c8-1]);} ;
                }
              }
            }
            if ((c1 == c3) && (c4 == c6) && (ny >= 2)) {
              for (c7=max(max(max(32*c5,32*c6-nx+32),0),32*c6);c7<=min(min(tmax-1,32*c5-ny+31),32*c6+30);c7++) {
                {ey[0][0]=c7;} ;
                for (c9=c7+1;c9<=32*c6+31;c9++) {
                  {ey[-c7+c9][0]=ey[-c7+c9][0]-((double)(1))/2*(hz[-c7+c9][0]-hz[-c7+c9-1][0]);} ;
                }
                for (c8=c7+1;c8<=c7+ny-1;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  for (c9=c7+1;c9<=32*c6+31;c9++) {
                    {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                    {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                    {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                  }
                }
                for (c9=c7+1;c9<=32*c6+31;c9++) {
                  {hz[-c7+c9-1][ny-1]=hz[-c7+c9-1][ny-1]-((double)(7))/10*(ey[1+-c7+c9-1][ny-1]+ex[-c7+c9-1][1+ny-1]-ex[-c7+c9-1][ny-1]-ey[-c7+c9-1][ny-1]);} ;
                }
              }
            }
            if ((c1 == c3) && (c4 == c6)) {
              for (c7=max(max(max(0,32*c6),32*c6-nx+32),32*c5-ny+32);c7<=min(min(tmax-1,32*c6+30),32*c5-1);c7++) {
                for (c8=32*c5;c8<=32*c5+31;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  for (c9=c7+1;c9<=32*c6+31;c9++) {
                    {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                    {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                    {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                  }
                }
              }
            }
            if ((c1 == c3) && (c4 == c6)) {
              for (c7=max(max(max(max(32*c5,32*c6-nx+32),0),32*c6),32*c5-ny+32);c7<=min(min(tmax-1,32*c5+30),32*c6+30);c7++) {
                {ey[0][0]=c7;} ;
                for (c9=c7+1;c9<=32*c6+31;c9++) {
                  {ey[-c7+c9][0]=ey[-c7+c9][0]-((double)(1))/2*(hz[-c7+c9][0]-hz[-c7+c9-1][0]);} ;
                }
                for (c8=c7+1;c8<=32*c5+31;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  for (c9=c7+1;c9<=32*c6+31;c9++) {
                    {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                    {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                    {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                  }
                }
              }
            }
            if ((c1 == c3) && (c4 == c6) && (nx >= 2) && (ny == 1)) {
              for (c7=max(max(0,32*c6),32*c5);c7<=min(min(tmax-1,32*c6+30),32*c5+30);c7++) {
                {ey[0][0]=c7;} ;
                for (c9=c7+1;c9<=min(c7+nx-1,32*c6+31);c9++) {
                  {ey[-c7+c9][0]=ey[-c7+c9][0]-((double)(1))/2*(hz[-c7+c9][0]-hz[-c7+c9-1][0]);} ;
                }
                for (c9=c7+1;c9<=min(c7+nx,32*c6+31);c9++) {
                  {hz[-c7+c9-1][0]=hz[-c7+c9-1][0]-((double)(7))/10*(ey[1+-c7+c9-1][0]+ex[-c7+c9-1][1+0]-ex[-c7+c9-1][0]-ey[-c7+c9-1][0]);} ;
                }
              }
            }
            if ((c1 == c3) && (c4 == c6) && (nx == 1)) {
              for (c7=max(max(0,32*c6),32*c5-ny+1);c7<=min(min(min(32*c5-1,tmax-1),32*c5-ny+31),32*c6+30);c7++) {
                for (c8=32*c5;c8<=c7+ny-1;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  {hz[0][-c7+c8-1]=hz[0][-c7+c8-1]-((double)(7))/10*(ey[1+0][-c7+c8-1]+ex[0][1+-c7+c8-1]-ex[0][-c7+c8-1]-ey[0][-c7+c8-1]);} ;
                }
                {hz[0][ny-1]=hz[0][ny-1]-((double)(7))/10*(ey[1+0][ny-1]+ex[0][1+ny-1]-ex[0][ny-1]-ey[0][ny-1]);} ;
              }
            }
            if ((c1 == c3) && (c4 == c6) && (nx == 1) && (ny >= 2)) {
              for (c7=max(max(32*c5,0),32*c6);c7<=min(min(tmax-1,32*c5-ny+31),32*c6+30);c7++) {
                {ey[0][0]=c7;} ;
                for (c8=c7+1;c8<=c7+ny-1;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  {hz[0][-c7+c8-1]=hz[0][-c7+c8-1]-((double)(7))/10*(ey[1+0][-c7+c8-1]+ex[0][1+-c7+c8-1]-ex[0][-c7+c8-1]-ey[0][-c7+c8-1]);} ;
                }
                {hz[0][ny-1]=hz[0][ny-1]-((double)(7))/10*(ey[1+0][ny-1]+ex[0][1+ny-1]-ex[0][ny-1]-ey[0][ny-1]);} ;
              }
            }
            if ((c1 == c3) && (c4 == c6) && (nx == 1)) {
              for (c7=max(max(0,32*c6),32*c5-ny+32);c7<=min(min(tmax-1,32*c6+30),32*c5-1);c7++) {
                for (c8=32*c5;c8<=32*c5+31;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  {hz[0][-c7+c8-1]=hz[0][-c7+c8-1]-((double)(7))/10*(ey[1+0][-c7+c8-1]+ex[0][1+-c7+c8-1]-ex[0][-c7+c8-1]-ey[0][-c7+c8-1]);} ;
                }
              }
            }
            if ((c1 == c3) && (c4 == c6) && (nx == 1)) {
              for (c7=max(max(max(32*c5,0),32*c6),32*c5-ny+32);c7<=min(min(tmax-1,32*c5+30),32*c6+30);c7++) {
                {ey[0][0]=c7;} ;
                for (c8=c7+1;c8<=32*c5+31;c8++) {
                  {ey[0][-c7+c8]=c7;} ;
                  {ex[0][-c7+c8]=ex[0][-c7+c8]-((double)(1))/2*(hz[0][-c7+c8]-hz[0][-c7+c8-1]);} ;
                  {hz[0][-c7+c8-1]=hz[0][-c7+c8-1]-((double)(7))/10*(ey[1+0][-c7+c8-1]+ex[0][1+-c7+c8-1]-ex[0][-c7+c8-1]-ey[0][-c7+c8-1]);} ;
                }
              }
            }
            if ((c1 == c3) && (c4 == c6) && (nx == 1) && (ny == 1)) {
              for (c7=max(max(0,32*c6),32*c5);c7<=min(min(tmax-1,32*c6+30),32*c5+30);c7++) {
                {ey[0][0]=c7;} ;
                {hz[0][0]=hz[0][0]-((double)(7))/10*(ey[1+0][0]+ex[0][1+0]-ex[0][0]-ey[0][0]);} ;
              }
            }
            for (c7=max(max(max(32*c6-nx+1,0),32*c5-ny+1),32*c4);c7<=min(min(min(min(min(32*c5-1,32*c6-nx+31),32*c6-1),tmax-1),32*c4+31),32*c5-ny+31);c7++) {
              for (c8=32*c5;c8<=c7+ny-1;c8++) {
                for (c9=32*c6;c9<=c7+nx-1;c9++) {
                  {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                  {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                  {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                }
                {hz[nx-1][-c7+c8-1]=hz[nx-1][-c7+c8-1]-((double)(7))/10*(ey[1+nx-1][-c7+c8-1]+ex[nx-1][1+-c7+c8-1]-ex[nx-1][-c7+c8-1]-ey[nx-1][-c7+c8-1]);} ;
              }
              for (c9=32*c6;c9<=c7+nx;c9++) {
                {hz[-c7+c9-1][ny-1]=hz[-c7+c9-1][ny-1]-((double)(7))/10*(ey[1+-c7+c9-1][ny-1]+ex[-c7+c9-1][1+ny-1]-ex[-c7+c9-1][ny-1]-ey[-c7+c9-1][ny-1]);} ;
              }
            }
            if (ny >= 2) {
              for (c7=max(max(max(32*c5,32*c6-nx+1),0),32*c4);c7<=min(min(min(min(32*c6-nx+31,32*c6-1),tmax-1),32*c4+31),32*c5-ny+31);c7++) {
                for (c9=32*c6;c9<=c7+nx-1;c9++) {
                  {ey[-c7+c9][0]=ey[-c7+c9][0]-((double)(1))/2*(hz[-c7+c9][0]-hz[-c7+c9-1][0]);} ;
                }
                for (c8=c7+1;c8<=c7+ny-1;c8++) {
                  for (c9=32*c6;c9<=c7+nx-1;c9++) {
                    {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                    {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                    {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                  }
                  {hz[nx-1][-c7+c8-1]=hz[nx-1][-c7+c8-1]-((double)(7))/10*(ey[1+nx-1][-c7+c8-1]+ex[nx-1][1+-c7+c8-1]-ex[nx-1][-c7+c8-1]-ey[nx-1][-c7+c8-1]);} ;
                }
                for (c9=32*c6;c9<=c7+nx;c9++) {
                  {hz[-c7+c9-1][ny-1]=hz[-c7+c9-1][ny-1]-((double)(7))/10*(ey[1+-c7+c9-1][ny-1]+ex[-c7+c9-1][1+ny-1]-ex[-c7+c9-1][ny-1]-ey[-c7+c9-1][ny-1]);} ;
                }
              }
            }
            for (c7=max(max(max(0,32*c4),32*c6-nx+1),32*c5-ny+32);c7<=min(min(min(min(32*c6-1,tmax-1),32*c4+31),32*c5-1),32*c6-nx+31);c7++) {
              for (c8=32*c5;c8<=32*c5+31;c8++) {
                for (c9=32*c6;c9<=c7+nx-1;c9++) {
                  {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                  {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                  {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                }
                {hz[nx-1][-c7+c8-1]=hz[nx-1][-c7+c8-1]-((double)(7))/10*(ey[1+nx-1][-c7+c8-1]+ex[nx-1][1+-c7+c8-1]-ex[nx-1][-c7+c8-1]-ey[nx-1][-c7+c8-1]);} ;
              }
            }
            for (c7=max(max(max(32*c6-nx+32,0),32*c5-ny+1),32*c4);c7<=min(min(min(min(32*c5-1,32*c6-1),tmax-1),32*c4+31),32*c5-ny+31);c7++) {
              for (c8=32*c5;c8<=c7+ny-1;c8++) {
                for (c9=32*c6;c9<=32*c6+31;c9++) {
                  {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                  {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                  {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                }
              }
              for (c9=32*c6;c9<=32*c6+31;c9++) {
                {hz[-c7+c9-1][ny-1]=hz[-c7+c9-1][ny-1]-((double)(7))/10*(ey[1+-c7+c9-1][ny-1]+ex[-c7+c9-1][1+ny-1]-ex[-c7+c9-1][ny-1]-ey[-c7+c9-1][ny-1]);} ;
              }
            }
            for (c7=max(max(max(max(32*c5,0),32*c4),32*c6-nx+1),32*c5-ny+32);c7<=min(min(min(min(32*c6-1,tmax-1),32*c5+30),32*c4+31),32*c6-nx+31);c7++) {
              for (c9=32*c6;c9<=c7+nx-1;c9++) {
                {ey[-c7+c9][0]=ey[-c7+c9][0]-((double)(1))/2*(hz[-c7+c9][0]-hz[-c7+c9-1][0]);} ;
              }
              for (c8=c7+1;c8<=32*c5+31;c8++) {
                for (c9=32*c6;c9<=c7+nx-1;c9++) {
                  {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                  {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                  {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                }
                {hz[nx-1][-c7+c8-1]=hz[nx-1][-c7+c8-1]-((double)(7))/10*(ey[1+nx-1][-c7+c8-1]+ex[nx-1][1+-c7+c8-1]-ex[nx-1][-c7+c8-1]-ey[nx-1][-c7+c8-1]);} ;
              }
            }
            if (ny >= 2) {
              for (c7=max(max(max(32*c5,32*c6-nx+32),0),32*c4);c7<=min(min(min(32*c6-1,tmax-1),32*c4+31),32*c5-ny+31);c7++) {
                for (c9=32*c6;c9<=32*c6+31;c9++) {
                  {ey[-c7+c9][0]=ey[-c7+c9][0]-((double)(1))/2*(hz[-c7+c9][0]-hz[-c7+c9-1][0]);} ;
                }
                for (c8=c7+1;c8<=c7+ny-1;c8++) {
                  for (c9=32*c6;c9<=32*c6+31;c9++) {
                    {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                    {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                    {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                  }
                }
                for (c9=32*c6;c9<=32*c6+31;c9++) {
                  {hz[-c7+c9-1][ny-1]=hz[-c7+c9-1][ny-1]-((double)(7))/10*(ey[1+-c7+c9-1][ny-1]+ex[-c7+c9-1][1+ny-1]-ex[-c7+c9-1][ny-1]-ey[-c7+c9-1][ny-1]);} ;
                }
              }
            }
/*@ begin Loop(
  for (c7=max(max(max(0,32*c4),32*c6-nx+32),32*c5-ny+32);c7<=min(min(min(32*c6-1,tmax-1),32*c4+31),32*c5-1);c7++) {
  transform UnrollJam(ufactor=4)
  for (c8=32*c5;c8<=32*c5+31;c8++) {
  transform UnrollJam(ufactor=4)
  for (c9=32*c6;c9<=32*c6+31;c9++) {
   ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);
   ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);
   hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);
  }
  }
  }
) @*/for (c7=max(max(max(0,32*c4),32*c6-nx+32),32*c5-ny+32); c7<=min(min(min(32*c6-1,tmax-1),32*c4+31),32*c5-1); c7++ ) {
  {
    for (c8=32*c5; c8<=32*c5+28; c8=c8+4) {
      for (c9=32*c6; c9<=32*c6+28; c9=c9+4) {
        ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);
        ey[-c7+c9][-c7+c8+1]=ey[-c7+c9][-c7+c8+1]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+1]-hz[-c7+c9-1][-c7+c8+1]);
        ey[-c7+c9][-c7+c8+2]=ey[-c7+c9][-c7+c8+2]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+2]-hz[-c7+c9-1][-c7+c8+2]);
        ey[-c7+c9][-c7+c8+3]=ey[-c7+c9][-c7+c8+3]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+3]-hz[-c7+c9-1][-c7+c8+3]);
        ey[-c7+c9+1][-c7+c8]=ey[-c7+c9+1][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+1][-c7+c8]-hz[-c7+c9][-c7+c8]);
        ey[-c7+c9+1][-c7+c8+1]=ey[-c7+c9+1][-c7+c8+1]-0.5*((double)(1))*(hz[-c7+c9+1][-c7+c8+1]-hz[-c7+c9][-c7+c8+1]);
        ey[-c7+c9+1][-c7+c8+2]=ey[-c7+c9+1][-c7+c8+2]-0.5*((double)(1))*(hz[-c7+c9+1][-c7+c8+2]-hz[-c7+c9][-c7+c8+2]);
        ey[-c7+c9+1][-c7+c8+3]=ey[-c7+c9+1][-c7+c8+3]-0.5*((double)(1))*(hz[-c7+c9+1][-c7+c8+3]-hz[-c7+c9][-c7+c8+3]);
        ey[-c7+c9+2][-c7+c8]=ey[-c7+c9+2][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+2][-c7+c8]-hz[-c7+c9+1][-c7+c8]);
        ey[-c7+c9+2][-c7+c8+1]=ey[-c7+c9+2][-c7+c8+1]-0.5*((double)(1))*(hz[-c7+c9+2][-c7+c8+1]-hz[-c7+c9+1][-c7+c8+1]);
        ey[-c7+c9+2][-c7+c8+2]=ey[-c7+c9+2][-c7+c8+2]-0.5*((double)(1))*(hz[-c7+c9+2][-c7+c8+2]-hz[-c7+c9+1][-c7+c8+2]);
        ey[-c7+c9+2][-c7+c8+3]=ey[-c7+c9+2][-c7+c8+3]-0.5*((double)(1))*(hz[-c7+c9+2][-c7+c8+3]-hz[-c7+c9+1][-c7+c8+3]);
        ey[-c7+c9+3][-c7+c8]=ey[-c7+c9+3][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+3][-c7+c8]-hz[-c7+c9+2][-c7+c8]);
        ey[-c7+c9+3][-c7+c8+1]=ey[-c7+c9+3][-c7+c8+1]-0.5*((double)(1))*(hz[-c7+c9+3][-c7+c8+1]-hz[-c7+c9+2][-c7+c8+1]);
        ey[-c7+c9+3][-c7+c8+2]=ey[-c7+c9+3][-c7+c8+2]-0.5*((double)(1))*(hz[-c7+c9+3][-c7+c8+2]-hz[-c7+c9+2][-c7+c8+2]);
        ey[-c7+c9+3][-c7+c8+3]=ey[-c7+c9+3][-c7+c8+3]-0.5*((double)(1))*(hz[-c7+c9+3][-c7+c8+3]-hz[-c7+c9+2][-c7+c8+3]);
        ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);
        ex[-c7+c9][-c7+c8+1]=ex[-c7+c9][-c7+c8+1]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+1]-hz[-c7+c9][-c7+c8]);
        ex[-c7+c9][-c7+c8+2]=ex[-c7+c9][-c7+c8+2]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+2]-hz[-c7+c9][-c7+c8+1]);
        ex[-c7+c9][-c7+c8+3]=ex[-c7+c9][-c7+c8+3]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+3]-hz[-c7+c9][-c7+c8+2]);
        ex[-c7+c9+1][-c7+c8]=ex[-c7+c9+1][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+1][-c7+c8]-hz[-c7+c9+1][-c7+c8-1]);
        ex[-c7+c9+1][-c7+c8+1]=ex[-c7+c9+1][-c7+c8+1]-0.5*((double)(1))*(hz[-c7+c9+1][-c7+c8+1]-hz[-c7+c9+1][-c7+c8]);
        ex[-c7+c9+1][-c7+c8+2]=ex[-c7+c9+1][-c7+c8+2]-0.5*((double)(1))*(hz[-c7+c9+1][-c7+c8+2]-hz[-c7+c9+1][-c7+c8+1]);
        ex[-c7+c9+1][-c7+c8+3]=ex[-c7+c9+1][-c7+c8+3]-0.5*((double)(1))*(hz[-c7+c9+1][-c7+c8+3]-hz[-c7+c9+1][-c7+c8+2]);
        ex[-c7+c9+2][-c7+c8]=ex[-c7+c9+2][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+2][-c7+c8]-hz[-c7+c9+2][-c7+c8-1]);
        ex[-c7+c9+2][-c7+c8+1]=ex[-c7+c9+2][-c7+c8+1]-0.5*((double)(1))*(hz[-c7+c9+2][-c7+c8+1]-hz[-c7+c9+2][-c7+c8]);
        ex[-c7+c9+2][-c7+c8+2]=ex[-c7+c9+2][-c7+c8+2]-0.5*((double)(1))*(hz[-c7+c9+2][-c7+c8+2]-hz[-c7+c9+2][-c7+c8+1]);
        ex[-c7+c9+2][-c7+c8+3]=ex[-c7+c9+2][-c7+c8+3]-0.5*((double)(1))*(hz[-c7+c9+2][-c7+c8+3]-hz[-c7+c9+2][-c7+c8+2]);
        ex[-c7+c9+3][-c7+c8]=ex[-c7+c9+3][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+3][-c7+c8]-hz[-c7+c9+3][-c7+c8-1]);
        ex[-c7+c9+3][-c7+c8+1]=ex[-c7+c9+3][-c7+c8+1]-0.5*((double)(1))*(hz[-c7+c9+3][-c7+c8+1]-hz[-c7+c9+3][-c7+c8]);
        ex[-c7+c9+3][-c7+c8+2]=ex[-c7+c9+3][-c7+c8+2]-0.5*((double)(1))*(hz[-c7+c9+3][-c7+c8+2]-hz[-c7+c9+3][-c7+c8+1]);
        ex[-c7+c9+3][-c7+c8+3]=ex[-c7+c9+3][-c7+c8+3]-0.5*((double)(1))*(hz[-c7+c9+3][-c7+c8+3]-hz[-c7+c9+3][-c7+c8+2]);
        hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-0.1*((double)(7))*(ey[-c7+c9][-c7+c8-1]+ex[-c7+c9-1][-c7+c8]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);
        hz[-c7+c9-1][-c7+c8]=hz[-c7+c9-1][-c7+c8]-0.1*((double)(7))*(ey[-c7+c9][-c7+c8]+ex[-c7+c9-1][-c7+c8+1]-ex[-c7+c9-1][-c7+c8]-ey[-c7+c9-1][-c7+c8]);
        hz[-c7+c9-1][-c7+c8+1]=hz[-c7+c9-1][-c7+c8+1]-0.1*((double)(7))*(ey[-c7+c9][-c7+c8+1]+ex[-c7+c9-1][-c7+c8+2]-ex[-c7+c9-1][-c7+c8+1]-ey[-c7+c9-1][-c7+c8+1]);
        hz[-c7+c9-1][-c7+c8+2]=hz[-c7+c9-1][-c7+c8+2]-0.1*((double)(7))*(ey[-c7+c9][-c7+c8+2]+ex[-c7+c9-1][-c7+c8+3]-ex[-c7+c9-1][-c7+c8+2]-ey[-c7+c9-1][-c7+c8+2]);
        hz[-c7+c9][-c7+c8-1]=hz[-c7+c9][-c7+c8-1]-0.1*((double)(7))*(ey[-c7+c9+1][-c7+c8-1]+ex[-c7+c9][-c7+c8]-ex[-c7+c9][-c7+c8-1]-ey[-c7+c9][-c7+c8-1]);
        hz[-c7+c9][-c7+c8]=hz[-c7+c9][-c7+c8]-0.1*((double)(7))*(ey[-c7+c9+1][-c7+c8]+ex[-c7+c9][-c7+c8+1]-ex[-c7+c9][-c7+c8]-ey[-c7+c9][-c7+c8]);
        hz[-c7+c9][-c7+c8+1]=hz[-c7+c9][-c7+c8+1]-0.1*((double)(7))*(ey[-c7+c9+1][-c7+c8+1]+ex[-c7+c9][-c7+c8+2]-ex[-c7+c9][-c7+c8+1]-ey[-c7+c9][-c7+c8+1]);
        hz[-c7+c9][-c7+c8+2]=hz[-c7+c9][-c7+c8+2]-0.1*((double)(7))*(ey[-c7+c9+1][-c7+c8+2]+ex[-c7+c9][-c7+c8+3]-ex[-c7+c9][-c7+c8+2]-ey[-c7+c9][-c7+c8+2]);
        hz[-c7+c9+1][-c7+c8-1]=hz[-c7+c9+1][-c7+c8-1]-0.1*((double)(7))*(ey[-c7+c9+2][-c7+c8-1]+ex[-c7+c9+1][-c7+c8]-ex[-c7+c9+1][-c7+c8-1]-ey[-c7+c9+1][-c7+c8-1]);
        hz[-c7+c9+1][-c7+c8]=hz[-c7+c9+1][-c7+c8]-0.1*((double)(7))*(ey[-c7+c9+2][-c7+c8]+ex[-c7+c9+1][-c7+c8+1]-ex[-c7+c9+1][-c7+c8]-ey[-c7+c9+1][-c7+c8]);
        hz[-c7+c9+1][-c7+c8+1]=hz[-c7+c9+1][-c7+c8+1]-0.1*((double)(7))*(ey[-c7+c9+2][-c7+c8+1]+ex[-c7+c9+1][-c7+c8+2]-ex[-c7+c9+1][-c7+c8+1]-ey[-c7+c9+1][-c7+c8+1]);
        hz[-c7+c9+1][-c7+c8+2]=hz[-c7+c9+1][-c7+c8+2]-0.1*((double)(7))*(ey[-c7+c9+2][-c7+c8+2]+ex[-c7+c9+1][-c7+c8+3]-ex[-c7+c9+1][-c7+c8+2]-ey[-c7+c9+1][-c7+c8+2]);
        hz[-c7+c9+2][-c7+c8-1]=hz[-c7+c9+2][-c7+c8-1]-0.1*((double)(7))*(ey[-c7+c9+3][-c7+c8-1]+ex[-c7+c9+2][-c7+c8]-ex[-c7+c9+2][-c7+c8-1]-ey[-c7+c9+2][-c7+c8-1]);
        hz[-c7+c9+2][-c7+c8]=hz[-c7+c9+2][-c7+c8]-0.1*((double)(7))*(ey[-c7+c9+3][-c7+c8]+ex[-c7+c9+2][-c7+c8+1]-ex[-c7+c9+2][-c7+c8]-ey[-c7+c9+2][-c7+c8]);
        hz[-c7+c9+2][-c7+c8+1]=hz[-c7+c9+2][-c7+c8+1]-0.1*((double)(7))*(ey[-c7+c9+3][-c7+c8+1]+ex[-c7+c9+2][-c7+c8+2]-ex[-c7+c9+2][-c7+c8+1]-ey[-c7+c9+2][-c7+c8+1]);
        hz[-c7+c9+2][-c7+c8+2]=hz[-c7+c9+2][-c7+c8+2]-0.1*((double)(7))*(ey[-c7+c9+3][-c7+c8+2]+ex[-c7+c9+2][-c7+c8+3]-ex[-c7+c9+2][-c7+c8+2]-ey[-c7+c9+2][-c7+c8+2]);
      }
      for (; c9<=32*c6+31; c9=c9+1) {
        ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);
        ey[-c7+c9][-c7+c8+1]=ey[-c7+c9][-c7+c8+1]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+1]-hz[-c7+c9-1][-c7+c8+1]);
        ey[-c7+c9][-c7+c8+2]=ey[-c7+c9][-c7+c8+2]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+2]-hz[-c7+c9-1][-c7+c8+2]);
        ey[-c7+c9][-c7+c8+3]=ey[-c7+c9][-c7+c8+3]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+3]-hz[-c7+c9-1][-c7+c8+3]);
        ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);
        ex[-c7+c9][-c7+c8+1]=ex[-c7+c9][-c7+c8+1]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+1]-hz[-c7+c9][-c7+c8]);
        ex[-c7+c9][-c7+c8+2]=ex[-c7+c9][-c7+c8+2]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+2]-hz[-c7+c9][-c7+c8+1]);
        ex[-c7+c9][-c7+c8+3]=ex[-c7+c9][-c7+c8+3]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8+3]-hz[-c7+c9][-c7+c8+2]);
        hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-0.1*((double)(7))*(ey[-c7+c9][-c7+c8-1]+ex[-c7+c9-1][-c7+c8]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);
        hz[-c7+c9-1][-c7+c8]=hz[-c7+c9-1][-c7+c8]-0.1*((double)(7))*(ey[-c7+c9][-c7+c8]+ex[-c7+c9-1][-c7+c8+1]-ex[-c7+c9-1][-c7+c8]-ey[-c7+c9-1][-c7+c8]);
        hz[-c7+c9-1][-c7+c8+1]=hz[-c7+c9-1][-c7+c8+1]-0.1*((double)(7))*(ey[-c7+c9][-c7+c8+1]+ex[-c7+c9-1][-c7+c8+2]-ex[-c7+c9-1][-c7+c8+1]-ey[-c7+c9-1][-c7+c8+1]);
        hz[-c7+c9-1][-c7+c8+2]=hz[-c7+c9-1][-c7+c8+2]-0.1*((double)(7))*(ey[-c7+c9][-c7+c8+2]+ex[-c7+c9-1][-c7+c8+3]-ex[-c7+c9-1][-c7+c8+2]-ey[-c7+c9-1][-c7+c8+2]);
      }
    }
    for (; c8<=32*c5+31; c8=c8+1) {
      {
        for (c9=32*c6; c9<=32*c6+28; c9=c9+4) {
          ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);
          ey[-c7+c9+1][-c7+c8]=ey[-c7+c9+1][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+1][-c7+c8]-hz[-c7+c9][-c7+c8]);
          ey[-c7+c9+2][-c7+c8]=ey[-c7+c9+2][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+2][-c7+c8]-hz[-c7+c9+1][-c7+c8]);
          ey[-c7+c9+3][-c7+c8]=ey[-c7+c9+3][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+3][-c7+c8]-hz[-c7+c9+2][-c7+c8]);
          ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);
          ex[-c7+c9+1][-c7+c8]=ex[-c7+c9+1][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+1][-c7+c8]-hz[-c7+c9+1][-c7+c8-1]);
          ex[-c7+c9+2][-c7+c8]=ex[-c7+c9+2][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+2][-c7+c8]-hz[-c7+c9+2][-c7+c8-1]);
          ex[-c7+c9+3][-c7+c8]=ex[-c7+c9+3][-c7+c8]-0.5*((double)(1))*(hz[-c7+c9+3][-c7+c8]-hz[-c7+c9+3][-c7+c8-1]);
          hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-0.1*((double)(7))*(ey[-c7+c9][-c7+c8-1]+ex[-c7+c9-1][-c7+c8]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);
          hz[-c7+c9][-c7+c8-1]=hz[-c7+c9][-c7+c8-1]-0.1*((double)(7))*(ey[-c7+c9+1][-c7+c8-1]+ex[-c7+c9][-c7+c8]-ex[-c7+c9][-c7+c8-1]-ey[-c7+c9][-c7+c8-1]);
          hz[-c7+c9+1][-c7+c8-1]=hz[-c7+c9+1][-c7+c8-1]-0.1*((double)(7))*(ey[-c7+c9+2][-c7+c8-1]+ex[-c7+c9+1][-c7+c8]-ex[-c7+c9+1][-c7+c8-1]-ey[-c7+c9+1][-c7+c8-1]);
          hz[-c7+c9+2][-c7+c8-1]=hz[-c7+c9+2][-c7+c8-1]-0.1*((double)(7))*(ey[-c7+c9+3][-c7+c8-1]+ex[-c7+c9+2][-c7+c8]-ex[-c7+c9+2][-c7+c8-1]-ey[-c7+c9+2][-c7+c8-1]);
        }
        for (; c9<=32*c6+31; c9=c9+1) {
          ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);
          ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);
          hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);
        }
      }
    }
  }
}
/*@ end @*/

            for (c7=max(max(max(max(32*c5,32*c6-nx+32),0),32*c4),32*c5-ny+32);c7<=min(min(min(32*c6-1,tmax-1),32*c5+30),32*c4+31);c7++) {
              for (c9=32*c6;c9<=32*c6+31;c9++) {
                {ey[-c7+c9][0]=ey[-c7+c9][0]-((double)(1))/2*(hz[-c7+c9][0]-hz[-c7+c9-1][0]);} ;
              }
              for (c8=c7+1;c8<=32*c5+31;c8++) {
                for (c9=32*c6;c9<=32*c6+31;c9++) {
                  {ey[-c7+c9][-c7+c8]=ey[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9-1][-c7+c8]);} ;
                  {ex[-c7+c9][-c7+c8]=ex[-c7+c9][-c7+c8]-((double)(1))/2*(hz[-c7+c9][-c7+c8]-hz[-c7+c9][-c7+c8-1]);} ;
                  {hz[-c7+c9-1][-c7+c8-1]=hz[-c7+c9-1][-c7+c8-1]-((double)(7))/10*(ey[1+-c7+c9-1][-c7+c8-1]+ex[-c7+c9-1][1+-c7+c8-1]-ex[-c7+c9-1][-c7+c8-1]-ey[-c7+c9-1][-c7+c8-1]);} ;
                }
              }
            }
            if (ny == 1) {
              for (c7=max(max(max(0,32*c6-nx+1),32*c5),32*c4);c7<=min(min(min(32*c6-1,tmax-1),32*c4+31),32*c5+30);c7++) {
                for (c9=32*c6;c9<=min(c7+nx-1,32*c6+31);c9++) {
                  {ey[-c7+c9][0]=ey[-c7+c9][0]-((double)(1))/2*(hz[-c7+c9][0]-hz[-c7+c9-1][0]);} ;
                }
                for (c9=32*c6;c9<=min(c7+nx,32*c6+31);c9++) {
                  {hz[-c7+c9-1][0]=hz[-c7+c9-1][0]-((double)(7))/10*(ey[1+-c7+c9-1][0]+ex[-c7+c9-1][1+0]-ex[-c7+c9-1][0]-ey[-c7+c9-1][0]);} ;
                }
              }
            }
            if ((c1 == c3) && (c4 == c6) && (c5 <= min(floord(tmax-32,32),floord(32*c6-1,32))) && (nx >= 2) && (ny == 1)) {
              {ey[0][0]=32*c5+31;} ;
              for (c9=32*c5+32;c9<=min(32*c5+nx+30,32*c6+31);c9++) {
                {ey[-32*c5+c9-31][0]=ey[-32*c5+c9-31][0]-((double)(1))/2*(hz[-32*c5+c9-31][0]-hz[-32*c5+c9-31 -1][0]);} ;
              }
            }
            if ((c1 == c3) && (c4 == c6) && (c5 <= min(floord(tmax-32,32),floord(32*c6-1,32))) && (nx >= 2) && (ny >= 2)) {
              {ey[0][0]=32*c5+31;} ;
              for (c9=32*c5+32;c9<=min(32*c5+nx+30,32*c6+31);c9++) {
                {ey[-32*c5+c9-31][0]=ey[-32*c5+c9-31][0]-((double)(1))/2*(hz[-32*c5+c9-31][0]-hz[-32*c5+c9-31 -1][0]);} ;
              }
            }
            if ((c1 == c3) && (-c4 == -c6) && (c4 >= ceild(32*c5-31,32)) && (c4 <= min(floord(tmax-32,32),floord(32*c5-1,32))) && (nx == 1) && (ny >= 2)) {
              {ey[0][0]=32*c4+31;} ;
              for (c8=32*c4+32;c8<=min(32*c4+ny+30,32*c5+31);c8++) {
                {ey[0][-32*c4+c8-31]=32*c4+31;} ;
                {ex[0][-32*c4+c8-31]=ex[0][-32*c4+c8-31]-((double)(1))/2*(hz[0][-32*c4+c8-31]-hz[0][-32*c4+c8-31 -1]);} ;
              }
            }
            if ((c1 == c3) && (-c4 == -c6) && (c4 <= min(floord(tmax-32,32),c5-1)) && (nx == 1)) {
              for (c8=32*c5;c8<=min(32*c4+ny+30,32*c5+31);c8++) {
                {ey[0][-32*c4+c8-31]=32*c4+31;} ;
                {ex[0][-32*c4+c8-31]=ex[0][-32*c4+c8-31]-((double)(1))/2*(hz[0][-32*c4+c8-31]-hz[0][-32*c4+c8-31 -1]);} ;
              }
            }
            if ((c1 == c3) && (-c4 == -c6) && (c4 <= min(floord(tmax-32,32),c5)) && (nx == 1) && (ny == 1)) {
              {ey[0][0]=32*c4+31;} ;
            }
            if ((c1 == c3) && (c4 == c6) && (c5 <= min(floord(tmax-32,32),floord(32*c6-1,32))) && (nx == 1) && (ny == 1)) {
              {ey[0][0]=32*c5+31;} ;
            }
            if ((c1 == c3) && (c4 == c6) && (c5 <= min(floord(tmax-32,32),c6)) && (nx == 1) && (ny >= 2)) {
              {ey[0][0]=32*c5+31;} ;
            }
            if ((c1 == c3) && (-c4 == -c6) && (c4 >= ceild(32*c5-31,32)) && (c4 <= min(floord(tmax-32,32),floord(32*c5-1,32))) && (nx >= 2) && (ny >= 2)) {
              {ey[0][0]=32*c4+31;} ;
              for (c8=32*c4+32;c8<=min(32*c4+ny+30,32*c5+31);c8++) {
                {ey[0][-32*c4+c8-31]=32*c4+31;} ;
                {ex[0][-32*c4+c8-31]=ex[0][-32*c4+c8-31]-((double)(1))/2*(hz[0][-32*c4+c8-31]-hz[0][-32*c4+c8-31 -1]);} ;
              }
            }
            if ((c1 == c3) && (-c4 == -c6) && (c4 <= min(floord(tmax-32,32),c5-1)) && (nx >= 2)) {
              for (c8=32*c5;c8<=min(32*c4+ny+30,32*c5+31);c8++) {
                {ey[0][-32*c4+c8-31]=32*c4+31;} ;
                {ex[0][-32*c4+c8-31]=ex[0][-32*c4+c8-31]-((double)(1))/2*(hz[0][-32*c4+c8-31]-hz[0][-32*c4+c8-31 -1]);} ;
              }
            }
            if ((c1 == c3) && (-c4 == -c6) && (c4 <= min(floord(tmax-32,32),c5)) && (nx >= 2) && (ny == 1)) {
              {ey[0][0]=32*c4+31;} ;
            }
            if ((c1 == c3) && (-c4 == -c5) && (-c4 == -c6) && (c4 <= floord(tmax-32,32)) && (nx >= 2) && (ny >= 2)) {
              {ey[0][0]=32*c4+31;} ;
            }
            if ((c4 >= c5) && (c5 <= min(c6-1,floord(tmax-32,32))) && (ny == 1)) {
              for (c9=32*c6;c9<=min(32*c5+nx+30,32*c6+31);c9++) {
                {ey[-32*c5+c9-31][0]=ey[-32*c5+c9-31][0]-((double)(1))/2*(hz[-32*c5+c9-31][0]-hz[-32*c5+c9-31 -1][0]);} ;
              }
            }
            if ((c4 >= c5) && (c5 <= min(c6-1,floord(tmax-32,32))) && (ny >= 2)) {
              for (c9=32*c6;c9<=min(32*c5+nx+30,32*c6+31);c9++) {
                {ey[-32*c5+c9-31][0]=ey[-32*c5+c9-31][0]-((double)(1))/2*(hz[-32*c5+c9-31][0]-hz[-32*c5+c9-31 -1][0]);} ;
              }
            }
          }
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
                                    
