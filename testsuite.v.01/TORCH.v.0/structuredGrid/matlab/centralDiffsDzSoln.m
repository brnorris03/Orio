function dzFn = centralDiffsDzSoln(type,toughness)
% 
%   Solution for function output of central differences.
% 
%   Input:
%   int type   Type number for function to return.
%   double toughness  Difficulty parameter. Larger values result in more numerically challenging computations.
% 
%   Output:
%   function handle dzFn  The value of Dz at that point.
% 
% 
%   Alex Kaiser, LBNL, 9/2010
%
% 

if(type == 0)
    dzFn = @(x,y,z)sin(toughness * x) * sin(2.0 * toughness * y) * 4.0 * toughness * cos(4.0 * toughness * z);

elseif(type == 1)
    dzFn = @(x,y,z)toughness*x*y + 2*z*(toughness*x*y)^2 + 3*z*z*(toughness*x*y)^3 ;

elseif(type == 2)
    dzFn = @(x,y,z)toughness*x*y * exp(toughness*x*y*z) ;

else
    dzFn = @(x,y,z)sin(toughness * x) * sin(2.0 * toughness * y) * 4.0 * toughness * cos(4.0 * toughness * z);     
end
    