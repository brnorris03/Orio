function [x, d, q, K, primes] = seed(n, Nmax, Dimension)
%
% 	 Calculate the seed for random number generator.
% 	 Generates vectors of random points. 
% 	 
% 	 Input: 
% 		int n					Starting value of integers to generate. 
% 		int Nmax				Maximum number of points to generate. 
% 		int dimension			Dimension of random numbers to generate. 
% 	 
% 	 Output:
% 		double vector x         Current random vector. 
% 		double matrix d			2d array for "odometer constants".									
% 		double matrix q			Address of 2d array for inverses of integer powers.
% 		double k                Parameter. 
%       vector primes           Array of first prime numbers of length 'dimension' 
%   
%
%
%   Alex Kaiser, LBNL, 9/2010. 
%


%allocation 
K = zeros(Dimension,1); 
primes = getFirstNPrimes(Dimension); 
%precompute K vector to get array dimensions
for i=1:Dimension
    K(i) = ceil( log(Nmax + 1) / log(primes(i))) ; 
end

q = zeros(Dimension, max(K(:)) );  
d = zeros(Dimension, max(K(:)) ); 
x = zeros(Dimension, 1); 

for i=1:Dimension
    % k is already computed
    
    k = n; 
    x(i) = 0; 
    
    for j = 1:K(i)
        if j == 1
            q(i,j) = 1.0 / primes(i) ;
        else
            q(i,j) = q(i,j-1) / primes(i) ;
        end
        d(i,j) = mod(k, primes(i)) ; 
        k = (k - d(i,j)) / primes(i) ; 
        x(i) = x(i) + d(i,j) * q(i,j) ; 
    end    

end

