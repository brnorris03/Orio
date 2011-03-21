function soln = solve3DHeatMultiGrid(dt, tSteps, l, h, depth, maxItDown, maxItUp)

%   3D heat equation solve for fixed initial conditions. 
%   Multigrid, implicit method. 
% 
% 
%   input:
% 
%      double dt          Time step for solve 
%      int tSteps         total number of timesteps to perform
%      double l           width of cube, same on all dimensions  
%      double h           grid height
%      int depth          Number of steps to descend in V-cycle. 
% 	   int maxItDown      Maximum number of iterations to perform on down cycle.
%      int maxItUp        Maximum number of iterations to perform on up cycle.
% 	  
% 
%   output:
% 
%     soln    4D real     solution to heat equation. indexed (x,y,z,t)
% 
%   
%   Alex Kaiser, LBNL, 7/2010



%use fixed initial conditions 
f = @(x,y,z) sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z);


lambda = dt / (h*h) ;
N = l/h - 1;  %don't include ghost zones in this number
M = tSteps ; 
soln = zeros(N+2,N+2,N+2, M); 


%set initial conditions
%ghost zones added automatically in init function 
initTemp = initGrid3D(f, h, l, l, l); 
 
% leave ghost zones, as they are needed for Laplacian
soln(:,:,:,1) = initTemp ; 

for j = 2:tSteps    
   fprintf(1, '\nStep %d\n\n', j);  
   soln(:,:,:,j) = multiGridVCycle(soln(:,:,:,j-1), N, lambda, h, dt, soln(:,:,:,j-1), depth, maxItDown, maxItUp); 
end

% remove ghost zones for comparisons    
soln = soln(2:N+1, 2:N+1, 2:N+1, :) ; 


