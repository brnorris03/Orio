function [x, d] = mcRandom(x, d, q, K, primes, Dimension)
%
%
%   Calculate the seed for random number generator.
%	Generates vectors of random points. 
%
%   Input: 
% 		double vector x         Current random vector. 
% 		double matrix d			2d array for "odometer constants".									
% 		double matrix q			Address of 2d array for inverses of integer powers.
% 		double k                Parameter. 
%       vector primes           Array of first prime numbers of length 'dimension' 
%
% 	 Output:
% 		double vector x         Current random vector. 
% 		double matrix d			2d array for "odometer constants".	
%
%
%   Alex Kaiser, LBNL, 9/2010. 
%


for i=1:Dimension
    
    for j = 1:K(i)
       d(i,j) = d(i,j) + 1; 
       x(i) = x(i) + q(i,j); 
       
       if d(i,j) < primes(i)
           break
       end
       
       d(i,j) = 0; 
       
       if j == 1
           x(i) = x(i) - 1.0 ; 
       else
           x(i) = x(i) - q(i,j-1) ;
       end
       
    end
    
end
