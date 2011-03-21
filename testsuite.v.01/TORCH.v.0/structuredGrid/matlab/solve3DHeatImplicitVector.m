function soln = solve3DHeatImplicitVector(dt, tSteps, l, h)

% 3D heat equation solve for fixed initial conditions. 
% Implicit method using sparse matrices and sparse marix vector multiply
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


%use fixed initial conditions for now
f = @(x,y,z) sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z);


lambda = dt / (h*h) ;

N = l/h - 1;  %don't include ghost zones in this number
M = tSteps ; 
nCubed = N*N*N ; 
soln = zeros( nCubed , M); % don't include the ghost zones here


%set initial conditions
%ghost zones added automatically in init function 
initTemp = initGrid3D(f, h, l, l, l); 

%remove ghost zones, as they are zero and not needed
initTemp = initTemp(2:N+1, 2:N+1, 2:N+1); 
soln(:,1) = initTemp(:)  ; 


%initialize matrix
[rowPtr columnIndices values] = get3DHeatEqnMatrixImplicit(N, lambda); 

maxIt = length(soln(:,1)); %shouldn't take this long but want it to converge

for j = 2:tSteps    
   soln(:,j) = conjugateGradient(soln(:,j-1), nCubed, rowPtr, columnIndices, values, soln(:,j-1), maxIt); 
   j
end
 
% return to 3d array for easy comparisons
soln = reshape(soln, N, N, N, tSteps); 


