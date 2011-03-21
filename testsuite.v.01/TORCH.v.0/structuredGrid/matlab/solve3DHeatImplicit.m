function soln = solve3DHeatImplicit(dt, tSteps, l, h)

%   3D heat equation solve for fixed initial conditions. 
%   Explicit, stencil based method. 
% 
% 
%   input:
% 
%      double dt          Time step for solve 
%      int tSteps         total number of timesteps to perform
%      double l           width of cube, same on all dimensions  
%      double h           grid height
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
nCubed = N*N*N ; 
soln = zeros(N+2,N+2,N+2, M); 


%set initial conditions
%ghost zones added automatically in init function 
initTemp = initGrid3D(f, h, l, l, l); 

 
% leave ghost zones, as they are needed for Laplacian
soln(:,:,:,1) = initTemp ; 

maxIt = nCubed ; %nCubed; %shouldn't take this long but want it to converge

for j = 2:tSteps    
   soln(:,:,:,j) = conjugateGradientGrid(soln(:,:,:,j-1), soln(:,:,:,j-1), h, dt, maxIt); 
   j
end

% remove ghost zones for comparisons    
soln = soln(2:N+1, 2:N+1, 2:N+1, :) ; 





