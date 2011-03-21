function size = setSize(n, density)
%
% 	 Set global parameter size according to input.
% 	 Keep density constant.
% 
% 	 Input:
% 	 int n                      Number of particles
% 
% 	 Output:
% 	 double size (returned)     Global parameter for boundaries of region.
%      
% Alex Kaiser, LBNL, 10/2010
% 


    size = sqrt(density * n); 
end