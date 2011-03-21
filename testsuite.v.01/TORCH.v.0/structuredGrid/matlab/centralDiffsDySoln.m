function dyFn = centralDiffsDySoln(type,toughness)
% 
%   Solution for function output of central differences.
% 
%   Input:
%   int type   Type number for function to return.
%   double toughness  Difficulty parameter. Larger values result in more numerically challenging computations.
% 
%   Output:
%   function handle dyFn  The value of Dy at that point.
% 
% 
%   Alex Kaiser, LBNL, 9/2010
%
% 

if(type == 0)
    dyFn = @(x,y,z) sin(toughness * x) * 2.0 * toughness * cos(2.0 * toughness * y) * sin(4.0 * toughness * z);

elseif(type == 1)
    dyFn = @(x,y,z)toughness*x*z + 2*y*(toughness*x*z)^2 + 3*y*y*(toughness*x*z)^3 ;

elseif(type == 2)
    dyFn = @(x,y,z)toughness*x*z * exp(toughness*x*y*z) ;

else
    dyFn = @(x,y,z) sin(toughness * x) * 2.0 * toughness * cos(2.0 * toughness * y) * sin(4.0 * toughness * z);
end
    
   