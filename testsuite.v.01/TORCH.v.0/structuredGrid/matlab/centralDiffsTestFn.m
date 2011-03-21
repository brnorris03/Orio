function diffsFn = centralDiffsTestFn(type,toughness)
%   Test function for central differences.
%   Returns a function handle. 
% 
%   Input:
%   int type   Type number for function to return.
%   double toughness  Difficulty parameter. Larger values result in more numerically challenging computations.
% 
%   Output:
%   diffsFn     Function handle for evaluating central differences
% 
% 
%   Alex Kaiser, LBNL, 9/2010
%

if(type == 0)
    diffsFn = @(x,y,z) sin(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z);

elseif(type == 1)
    diffsFn = @(x,y,z) toughness*x*y*z + (toughness*x*y*z)^2 + (toughness*x*y*z)^3 ;

elseif(type == 2)
    diffsFn = @(x,y,z) exp(toughness*x*y*z) ;

else
    diffsFn = @(x,y,z) sin(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z);
end

