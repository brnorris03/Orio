#include <stdio.h>
#include <math.h>

#define MIN(X,Y) ((X) < (Y) ? : (X) : (Y))


int main(void)
{

  double x = 10;

  double y;

  y = sqrt(x);
  printf("%d\n", MIN(3,10));
  return 0;

} 


