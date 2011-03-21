function result = qMCintegrate(integrand, seedValue, points, Dimension, s)
%
%   Integrates the supplied function on the unit cube of the specified dimension. 
%	Uses quasi-monte Carlo integration. 
%
% 
% 	 Input:
% 			double	integrand(vector x, int n, double s)
% 								Integrand function. 
% 								Evaluates at x, a 'dimension' length double precision array.
% 								s - double precision parameter for integration function 
% 			int		seedValue	First index of random sequence to use. 	
% 			int		points		Number of points for which to evaluate function.
%           int		dimension	Dimension of problem. 
% 			double	s			Parameter for integration function. 
% 	 
% 	 Output:
% 			double result   	Integral of integrand on 'dimension' dimensional unit cube. 
%
%
%   Alex Kaiser, LBNL, 9/2010. 
%

[x, d, q, K, primes] = seed(seedValue, seedValue + points + 1, Dimension) ; 

result = 0; 
updateFreq = points / 10; %estimates printed every 'this many' steps

for j = 1:points
    [x, d] = mcRandom(x, d, q, K, primes, Dimension) ; 
    result = result + integrand(x, Dimension, s) ; 
    
    if( mod(j,updateFreq) == 0 )
        disp('steps completed = '); 
        j
        disp('current estimate = '); 
        result / j
    end
    
end

result = result / points; 


