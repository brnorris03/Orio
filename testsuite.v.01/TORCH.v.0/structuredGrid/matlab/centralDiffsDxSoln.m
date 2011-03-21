function dxFn = centralDiffsDxSoln(type,toughness)
% 
%   Solution for function output of central differences.
% 
%   Input:
%   int type   Type number for function to return.
%   double toughness  Difficulty parameter. Larger values result in more numerically challenging computations.
% 
%   Output:
%   function handle dxFn  The value of Dx at that point.
% 
% 
%   Alex Kaiser, LBNL, 9/2010
%
% 

if(type == 0)
    dxFn = @(x,y,z) toughness * cos(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z);

elseif(type == 1)
    dxFn = @(x,y,z) toughness*y*z + 2*x*(toughness*y*z)^2 + 3*x*x*(toughness*y*z)^3 ;

elseif(type == 2)
    dxFn = @(x,y,z) toughness*y*z * exp(toughness*x*y*z) ;

else
    dxFn = @(x,y,z) toughness * cos(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z);
end
    
    