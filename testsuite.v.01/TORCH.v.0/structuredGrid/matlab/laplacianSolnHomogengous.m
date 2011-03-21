function laplSolnHomogeneous = laplacianSolnHomogengous(type, toughness) 
%   Test function for homogeneous Laplacian.
%   Returns a function handle. 
% 
%   Input:
%   int type   Type number for function to laplFnHomogeneous = @(x,y,z).
%   double toughness  Difficulty parameter. Larger values result in more numerically challenging computations.
% 
%   Output:
%   diffsFn     Function handle for evaluating Laplaian. 
% 
% 
%   Alex Kaiser, LBNL, 9/2010
%


% select integer toughness to ensure correct boundary conditions.
toughness = round(toughness) ;
if(toughness < 1)
    toughness = 1;
end

if(type == 0)
    laplSolnHomogeneous = @(x,y,z)  -(3.0 * toughness * toughness * pi * pi) * ...
		                            0.025 * sin(toughness * pi * x) * ... 
                                            sin(pi * toughness * y) * ...
                                            sin(pi * toughness * z);
elseif(type == 1)
    laplSolnHomogeneous = @(x,y,z) 2 * toughness * ((x*x - x)*(y*y - y) + (x*x - x)*(z*z - z) + (y*y - y)*(z*z - z)) ;

else
    laplSolnHomogeneous = @(x,y,z) -(3.0 * toughness * toughness * pi * pi) * ...
		                            0.025 * sin(toughness * pi * x) * ... 
                                            sin(pi * toughness * y) * ...
                                            sin(pi * toughness * z);
end