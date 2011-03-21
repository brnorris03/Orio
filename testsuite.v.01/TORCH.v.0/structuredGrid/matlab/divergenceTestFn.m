function divFn = divergenceTestFn(type, toughness)
% 
%   Input functions for divergence. 
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
    divFn = @(x,y,z) [sin(toughness * x), sin(2.0 * toughness * y), sin(4.0 * toughness * z)];

elseif(type == 1)
    divFn = @(x,y,z) [toughness*x*y, toughness*x*y*z, -(toughness*y*z + 0.5 * toughness*x*z*z)] ;

elseif(type == 2)
    divFn = @(x,y,z) [exp(toughness * x), exp(2 * toughness * y), exp(4 * toughness * z)] ;

else
    divFn = @(x,y,z) [sin(toughness * x), sin(2.0 * toughness * y), sin(4.0 * toughness * z)];
end