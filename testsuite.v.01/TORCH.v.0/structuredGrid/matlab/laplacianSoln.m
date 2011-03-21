function laplFn = laplacianSoln(type, toughness)
% 
%   Solution for function output of Laplacian.
% 
%   Input:
%   int type   Type number for function to laplFn = @(x,y,z).
%   double toughness  Difficulty parameter. Larger values result in more numerically challenging computations.
% 
%   Output:
%   function handle laplFn  The value of Laplacian at that point.
% 
% 
%   Alex Kaiser, LBNL, 9/2010
%
% 

if(type == 0)
    laplFn = @(x,y,z) (-21.0 * toughness * toughness) * (sin(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z)) ;

elseif(type == 1)
    laplFn = @(x,y,z) (2.0*toughness*toughness) * (1.0 + 3.0*toughness*x*y*z) * (x*x*y*y + x*x*z*z + y*y*z*z) ;

elseif(type == 2)
    laplFn = @(x,y,z) (toughness * toughness * (x*x*y*y + x*x*z*z + y*y*z*z)) * exp(toughness*x*y*z) ;

else
    laplFn = @(x,y,z) (-21.0 * toughness * toughness) * (sin(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z)) ;

end