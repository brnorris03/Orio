function soln = solve3DHeat(dt, tSteps, l, h)

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




% use fixed initial conditions 
f = @(x,y,z) sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z);


lambda = dt / (h*h) ; 
if( lambda > (1.0/6.0))
    warning('Unstable grid parameters. Solution is unreliable.'); 
    lambda
end


N = l/h - 1;  %don't include ghost zones in this number
M = tSteps ; 
soln = zeros(N+2,N+2,N+2,M); %include the ghost zones here


% set initial conditions
% ghost zones added automatically in init function 
soln(:,:,:,1) = initGrid3D(f, h, l, l, l); 

for j = 2:tSteps
   soln(:,:,:,j) = dt * laplacian3D(soln(:,:,:,j-1), soln(:,:,:,j), h) + soln(:,:,:,j-1);     
end
 
% remove ghost zones for comparisons    
soln = soln(2:N+1, 2:N+1, 2:N+1, :) ; 




