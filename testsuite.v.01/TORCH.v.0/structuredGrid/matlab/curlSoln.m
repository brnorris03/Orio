function curlSol = curlSoln(type,toughness)
%   Test solution for curl.
%   Returns a function handle. 
% 
%   Input:
%   int type   Type number for function to return.
%   double toughness  Difficulty parameter. Larger values result in more numerically challenging computations.
% 
%   Output:
%   curlSol     Function handle for evaluating curl solution. 
% 
% 
%   Alex Kaiser, LBNL, 9/2010
%


if(type == 0)
    curlSol = @(x,y,z) [ ...
     4.0 * toughness * cos(4.0 * toughness * y) ;
           toughness * cos(      toughness * z) ;
     2.0 * toughness * cos(2.0 * toughness * x) ] ;

elseif(type == 1)
    % conservative field
    curlSol = @(x,y,z) [0,0,0]; 
    

elseif(type == 2)
    curlSol = @(x,y,z) [ ...
     4.0 * toughness * exp(4.0 * toughness * y) ...
          toughness * exp(      toughness * z) ...
     2.0 * toughness * exp(2.0 * toughness * x) ] ;

else
    curlSol = @(x,y,z) [ ...
     4.0 * toughness * cos(4.0 * toughness * y) ;
           toughness * cos(      toughness * z) ;
     2.0 * toughness * cos(2.0 * toughness * x) ] ;

end

    
