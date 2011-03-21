function gradFn = gradientSoln(type, toughness)
% 
%   Solution for function output of gradient.
% 
%   Input:
%   int type   Type number for function to return.
%   double toughness  Difficulty parameter. Larger values result in more numerically challenging computations.
% 
%   Output:
%   function handle gradFn  The value of Gradient at that point.
% 
% 
%   Alex Kaiser, LBNL, 9/2010
%
% 

if(type == 0)
    gradFn = @(x,y,z) [ toughness * cos(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z) , ...
                        sin(toughness * x) * 2.0 * toughness * cos(2.0 * toughness * y) * sin(4.0 * toughness * z) , ...
                        sin(toughness * x) * sin(2.0 * toughness * y) * 4.0 * toughness * cos(4.0 * toughness * z) ] ;

elseif(type == 1)
    gradFn = @(x,y,z) [ toughness*y*z + 2*x*(toughness*y*z)^2 + 3*x*x*(toughness*y*z)^3 , ...
                        toughness*x*z + 2*y*(toughness*x*z)^2 + 3*y*y*(toughness*x*z)^3 , ...
                        toughness*x*y + 2*z*(toughness*x*y)^2 + 3*z*z*(toughness*x*y)^3 ] ; 
elseif(type == 2)
    gradFn = @(x,y,z) [toughness*y*z * exp(toughness*x*y*z), ...
                       toughness*x*z * exp(toughness*x*y*z), ...
                       toughness*x*y * exp(toughness*x*y*z) ] ; 

else
    gradFn = @(x,y,z) [ toughness * cos(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z) , ...
                        sin(toughness * x) * 2.0 * toughness * cos(2.0 * toughness * y) * sin(4.0 * toughness * z) , ...
                        sin(toughness * x) * sin(2.0 * toughness * y) * 4.0 * toughness * cos(4.0 * toughness * z) ] ;
                    
end