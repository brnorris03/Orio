% 
%    Checks the 3D heat equation.
% 	 Spectral method.
% 
% 	 Parameters:
% 	 double dt          Time step.
% 	 int tSteps         Number of time steps to take.
% 	 double height      Grid height - Must be set such that there is a
%                           power of two number of internal points. 
%                           Set h = 1 / (2^n + 1) for desired n. 
% 	 double tolNorm     Tolerance for l2 relative error.
% 	 double tolMax      Tolerance for max difference.
%      
%      Alex Kaiser, LBNL, 7/2010


% parameters
dt = 0.001; 
tSteps = 5; 
l = 1.0; % width of cube, same on all dimensions  
h = 1.0 / (64.0 + 1);


N = l/h - 1;  %don't include ghost zones in this number
M = tSteps ; 


disp('Testing spectral method heat equation solve.');  


if( ~(exist('structuredGridPass', 'var')))
    structuredGridPass = true ; 
end

% link with FFT routines
currentLocation = pwd ; 
cd ../.. ; 
cd spectral_methods/matlab ; 
addpath(pwd) ; 
cd(currentLocation) ; 


%run numerical solution 
tic; 
soln = solve3DHeatSpectral(dt, tSteps, l, h);
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



