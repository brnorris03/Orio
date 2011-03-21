%    check qMCintegrate
% 
% 	 Computes the value of a Box integral using quasi-Monte Carlo integration. 
% 	 Also computes the result using an infinite series with provable bounds. 
% 	 Details of the integrals and series are described in the tech report. 
% 	 Compares the two results, taking the series result as true. 
% 	 
% 	 Parameters:
% 		int n               	Dimension of the function to integrate. 
% 		double s				Argument for box integral integrand function. 
% 		double tol				Tolerance for relative error in results. 
% 		int seedValue			Position in sequence at which to start random numbers. 
% 									Use in parallel compurations to obtain later portions 
% 									of the random number sequence. 
% 									Default = 0.
% 		int points				Number of points at which to evaluate integrand. 
% 
%
%   Alex Kaiser, LBNL, 9/2010. 
%


n = 6; 
s = 1; 

seedValue = 0; 
points = 1000000 ; 
tol = 10e-6; 

disp('Checking qMC integrate.'); 

disp('Dimension:'); 
n

tic
result = qMCintegrate(@boxIntegrand, seedValue, points, n, s)
toc


seriesVal = boxInt(n,s); 

if(seriesVal == 0.0)
    if(result == 0.0)
        relativeErr = 0.0
    else
        relativeErr = Inf; 
    end
else
    relativeErr = abs( (result - seriesVal) / seriesVal ) ; 
end

disp('relative error = ');
disp(relativeErr); 

if( relativeErr > tol )
    error('Relative error too high. Test failed.'); 
    qMCintegratePass = false ; 
else
    disp('Relative error within tolerance.'); 
    disp('Test passed.'); 
    qMCintegratePass = true ; 
end





