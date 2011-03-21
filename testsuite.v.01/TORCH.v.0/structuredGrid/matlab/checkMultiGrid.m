% 
%    Checks the 3D heat equation.
% 	 Multigrid, stencil based method.
% 
% 	 Parameters:
% 	 double dt          Time step.
% 	 int tSteps         Number of time steps to take.
% 	 double height      Grid height
%    int depth          Number of steps to descend in V-cycle. 
% 	 int maxItDown      Maximum number of iterations to perform on down cycle.
% 	 int maxItUp        Maximum number of iterations to perform on up cycle.
% 	 double tolNorm     Tolerance for l2 relative error.
% 	 double tolMax      Tolerance for max difference.
%      
%      Alex Kaiser, LBNL, 7/2010


disp('Begin MultiGrid tests.'); 

if( ~(exist('structuredGridPass', 'var')))
    structuredGridPass = true ; 
end


% parameters
dt = 0.001; 
tSteps = 5; 
l = 1.0; % width of cube, same on all dimensions  
h = 1.0 / 64.0;


N = l/h - 1;  %don't include ghost zones in this number
M = tSteps ; 

depth = 4; 
maxItDown = 15; 
maxItUp = 30; 

%run numerical solution 
tic; 
soln = solve3DHeatMultiGrid(dt, tSteps, l, h, depth, maxItDown, maxItUp);
fprintf(1,'\n\n'); 
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

