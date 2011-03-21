function pass = compareHeatEqnSoln(soln, tSteps, dt, h, l, tolNorm, tolMax)

% 
% 	 Checks whether numerical solution matches discretization of analytic solution.
% 	 Must match tolerances at all time steps to pass.
% 
% 	 Input:
% 	 gridArray numericalSolution  Approximation to the solution to check.
% 	 double tolNorm    Tolerance for l2 norm.
% 	 double tolMax     Tolerance for maximum difference.
% 
% 	 Output:
% 	 char pass (returned) Whether norms are under tolerances at all time steps.
% 
%      Alex Kaiser, LBNL, 7/2010
% 

pass = 1; 

N = l/h - 1;  %don't include ghost zones in this number
M = tSteps ; 

analyticSoln = zeros(N,N,N,M); % don't include the ghost zones here


t = 0; 
for step = 1:tSteps
    
    %set f to return the analytic solution to the known initial conditions 
    f = @(x,y,z) exp(-12*pi*pi*t) * (sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z));
    t = t + dt; 
    withGhosts = initGrid3D(f, h, l, l, l);  
    analyticSoln(:,:,:,step) = withGhosts(2:N+1, 2:N+1, 2:N+1); 
    
    step
        
    currentVector = soln(:,:,:,step); 
    analyticVector = analyticSoln(:,:,:,step);   
    
    relErr = norm( currentVector(:) - analyticVector(:)) / norm(analyticVector(:))
    
    if( relErr < tolNorm )
        disp('Heat equation relative error test passed at current time step.'); 
    else
        disp('Heat equation relative error test failed at current time step.'); 
        pass = 0; 
    end
    
    maxDiff = max(max(max(abs( soln(:,:,:,step) - analyticSoln(:,:,:,step) ))))  
    
    if( maxDiff < tolMax)
        disp('Heat equation max diff test passed at current time step.'); 
    else
        disp('Heat equation max diff test failed at current time step.'); 
        pass = 0; 
    end
    
end

