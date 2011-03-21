% checkAll.m
% 
% 	 Check script for two dimensional N Body computations.
% 
% 	 Default settings run each computation twice.
% 	 First, the naive and naive with cutoff are run with the same non-physical, repulsive force.
% 	 The RMS error between the forces computed with these methods is tested to be below tolerance.
% 	 Error between these forces is also evaluated.
% 
% 	 Parameters:
% 	 int n                       Number of particles to simulate.
% 	 int numStep                 Number of time steps to simulate.
% 	 int forceType               Which force to use.
% 	 	                             Type 0 = gravity-like attractive force.
% 	                                 Type 1 = repulsive.
% 	 char output                 If true, prints a log of positions of particles to a text file.
% 	                                 Default = false.
% 	 const int saveFreq          If (output), positions of particles will be saved every this many time steps.
% 	 char verify                 Selects whether force logging for verification will be performed.
% 	                                 Default = true.
%      
% Alex Kaiser, LBNL, 10/2010
% 


n = 50 ; 
numSteps = 15 ; 
forceType = 1 ; 
output = true ; 
saveFreq = 1 ; 
verify = true ; 

tol = 1e-4; 

disp('Begin N body tests.'); 


disp('Time with naive algorithm:') ; 

tic; 
naiveLog = runNaive(n,numSteps, forceType, output, saveFreq, verify) ; 
toc; 

disp('Time with cutoff algorithm:'); 
tic; 
cutoffLog = runCutoff(n,numSteps, forceType, output, saveFreq, verify) ;
toc; 

pass2d = verifyNBody(naiveLog, cutoffLog, tol) ; 

if pass2d
    disp('Two dimensional test passed'); 
end


tol = 1e-4; 

disp('Begin N body tests.'); 

disp('Time with naive algorithm:') ; 

tic; 
naiveLog = runNaive3d(n,numSteps, forceType, output, saveFreq, verify) ; 
toc; 

disp('Time with cutoff algorithm:'); 
tic; 
cutoffLog = runCutoff3d(n,numSteps, forceType, output, saveFreq, verify) ;
toc; 

pass3d = verifyNBody3d(naiveLog, cutoffLog, tol) ; 

if pass3d
    disp('Three dimensional test passed.'); 
end


if pass2d && pass3d
    disp('All tests passed.'); 
    nBodyPass = true ; 
else
    disp('Tests failed'); 
    nBodyPass = false ; 
end





