% 
%    Checks the 3D heat equation.
% 	 Explicit, matrix based method.
% 
% 	 Parameters:
% 	 double dt          Time step.
% 	 int tSteps         Number of time steps to take.
% 	 double height      Grid height
% 	 double tolNorm     Tolerance for l2 relative error.
% 	 double tolMax      Tolerance for max difference.
%      
%      Alex Kaiser, LBNL, 7/2010


% parameters
dt = 0.001; 
tSteps = 5; 
l = 1.0; % width of cube, same on all dimensions  
h = 1.0 / 64.0;


N = l/h - 1;  %don't include ghost zones in this number
M = tSteps ; 


% add path to sparse linear algebra directory and return here. 
currentLocation = pwd ; 
cd ../.. ; 
cd sparseLinearAlgebra/matlab ; 
addpath(pwd) ; 
cd(currentLocation) ; 

disp('Testing explicit heat equation solve, matrix version.');

if( ~(exist('structuredGridPass', 'var')))
    structuredGridPass = true ; 
end

%run numerical solution 
tic; 
soln = solve3DHeatVector(dt, tSteps, l, h); 
toc; 


tolNorm = 0.05; 
tolMax = 0.05; 


pass = compareHeatEqnSoln(soln, tSteps, dt, h, l, tolNorm, tolMax); 

if pass
    disp('Test passed.'); 
else
    disp('Test failed.');
end

structuredGridPass = structuredGridPass & pass ;

