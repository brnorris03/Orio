function soln = solve3DHeatVector(dt, tSteps, l, h)

% 3D heat equation solve for fixed initial conditions. 
% Explicit method using sparse matrices and sparse marix vector multiply
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
if( lambda > (1.0/6.0))
    warning('Unstable grid parameters. Solution is unreliable.'); 
    lambda
end

N = l/h - 1;  %don't include ghost zones in this number
M = tSteps ; 
nCubed = N*N*N ; 
soln = zeros( nCubed , M); % don't include the ghost zones here


%set initial conditions
%ghost zones added automatically in init function 
initTemp = initGrid3D(f, h, l, l, l); 

%remove ghost zones, as they are zero and not needed
initTemp = initTemp(2:N+1, 2:N+1, 2:N+1); 

soln(:,1) = initTemp(:); 

%initialize matrix
[rowPtr columnIndices values] = get3DHeatEqnMatrix(N, lambda); 

% use SpMv
%{
for j = 2:tSteps
   soln(:,j) = spmv(soln(:, j-1), nCubed, nCubed, rowPtr, columnIndices, values);      
end
%}

% Or use matrix powers
soln = matrixPowers(soln(:,1), nCubed, rowPtr, columnIndices, values, tSteps) ; 



% return to 3d array for comparisons
soln = reshape(soln, N, N, N, tSteps); 


