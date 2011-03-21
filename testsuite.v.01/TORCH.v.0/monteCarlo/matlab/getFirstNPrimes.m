function p = getFirstNPrimes(n)
%
% Returns an array of the first n prime numbers. 
% 
% Input:
%     int n       Number of primes to return. 
%     
% Output: 
%     p           Array of first n primes. 
%     
% 
%
% Alex Kaiser, LBNL, 9/2010
%


if n >= 15
    error('only first 15 primes supported'); 
end

primes = [2 3 5 7 11 13 17 19 23 29 31 37 41 43 47]; 

p = primes(1:n) ; 