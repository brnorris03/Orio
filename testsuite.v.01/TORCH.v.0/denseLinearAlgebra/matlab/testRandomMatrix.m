function [a lambdas] = testRandomMatrix(n, t)

% Generates randomized matrices with specified eigenvalue distribution. 
% First, generates eigenvalues and stores them in a diagonal matrix D. 
% Generates a randomized orthogonal matrix Q, then returns the product
%   a = Q' * D * Q. 
% 
% Source: http://www.netlib.org/lapack/lawnspdf/lawn182.pdf
% 
% Input: 
%   int n               Dimension of matrix returned. 
%   int t               Type number. 
%
% Output:
%   matrix a            Randomized matrix with known eigenvalues.  
%   vector lambdas      Eigenvalues of matrix a. 
%
%
% Alex Kaiser, LBNL, 9/2010



% Use epsilon at one for ulp. 
ulp = eps(); 

lambdas = zeros(n,1); 

if t == 1
    k = 1/ulp; 
    lambdas(1) = 1; 
    for i = 2:n
        lambdas(i) = 1/k; 
    end
    
elseif t == 2
    k = 1/ulp;  
    for i = 1:n-1
        lambdas(i) = 1; 
    end
    lambdas(n) = 1/k; 
    
elseif t == 3
    k = 1/ulp; 
    for i = 1:n
        lambdas(i) = k ^ (-(i-1)/(n-1)) ;         
    end
    
elseif t == 4
    k = 1/ulp; 
    for i = 1:n
        lambdas(i) = 1 - ((i-1)/(n-1)) * (1 - 1/k) ;        
    end
    
elseif t == 5
    lambdas = log(ulp).*rand(n,1);  % n random numbers, unif in [log(ulp), 0)
    lambdas = exp(lambdas); % their exponentials, which are dist on (1/k, 1)
    
    
elseif t == 6
    lambdas = rand(n,1);  %from uniform [0,1] dist
    
elseif t == 7
    for i = 1:n-1
        lambdas(i) = ulp * i ;
    end
    lambdas(n) = 1; 
    
elseif t == 8
    lambdas(1) = ulp; 
    for i = 2:n-1
        lambdas(i) = 1 + sqrt(ulp) * i; 
    end
    lambdas(n) = 2; 
    
elseif t == 9
    lambdas(1) = 1; 
    for i = 2:n
        lambdas(i) = lambdas(i-1) + 100*ulp; 
    end    
    
else
    error('unsupported type'); 
end

Q = getRandomOrthogonalMatrix(n);
D = diag(lambdas); 
a = Q' * D * Q; 

