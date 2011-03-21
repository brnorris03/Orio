function divSoln = divergenceSoln(type, toughness)
% 
%   Solution for function output of divergence.
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
    divSoln = @(x,y,z)toughness * cos(toughness * x) +  ... 
            2.0 * toughness * cos(2.0 * toughness * y) +  ...
            4.0 * toughness * cos(4.0 * toughness * z);

elseif(type == 1)
    divSoln = @(x,y,z)0.0 ;

elseif(type == 2)
    divSoln = @(x,y,z)toughness * (exp(toughness * x) + 2.0*exp(2.0*toughness*y) + 4.0*exp(4.0*toughness*z) ) ;

else
    divSoln = @(x,y,z)toughness * cos(toughness * x) +  ... 
            2.0 * toughness * cos(2.0 * toughness * y) +  ...
            4.0 * toughness * cos(4.0 * toughness * z);
end
    
    