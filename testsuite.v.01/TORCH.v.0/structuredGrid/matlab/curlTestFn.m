function curlFn = curlTestFn(type,toughness)
%   Test function for curl.
%   Returns a function handle. 
% 
%   Input:
%   int type   Type number for function to return.
%   double toughness  Difficulty parameter. Larger values result in more numerically challenging computations.
% 
%   Output:
%   curlFn     Function handle for evaluating curl. 
% 
% 
%   Alex Kaiser, LBNL, 9/2010
%

if(type == 0)
    curlFn = @(x,y,z) [ sin(toughness * z), sin(2.0 * toughness * x), sin(4.0 * toughness * y)] ;

elseif(type == 1)
    curlFn = @(x,y,z) [toughness*x*x*y*z, toughness * (z * x^3) / 3.0, toughness * (y * x^3) / 3.0] ;

elseif(type == 2)
    curlFn = @(x,y,z) [ exp(toughness * z), exp(2.0 * toughness * x), exp(4.0 * toughness * y) ] ;

else
    curlFn = @(x,y,z) [ sin(toughness * z), sin(2.0 * toughness * x), sin(4.0 * toughness * y)] ;
    
end



    