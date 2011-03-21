function [v beta] = householder(x)

% Computes the Householder Vector for use with QR algorithm for computing eigenvalues. 
%
% Upon completion, the following conditions hold about the matrix P, which 
% is not explicitly formed. 
%
%   P = I - Beta*v*v', where P is a Householder reflection matrix
%            P is orthagonal, v(1) = 1, 
%            Px = ||x|| * e_1
%            where ||x|| is the Euclidean norm, and e_1 is the first
%            element of the standard basis in R_n, [1,0 ... 0]
%
% Source: Golub and Van Loan, "Matrix Computations" p. 210
%
% Input: 
%   vector x            Source vector. 
%
% Output: 
%   vector v            Vector for computing Householder reflection.         
%   double beta         Coefficient for computing Householder reflection. 
%
%
% Alex Kaiser, LBNL, 9/2010


n = length(x); 
sigma = x(2:n)' * x(2:n); 
v = [1 ; x(2:n)] ; 

if (sigma == 0)
    beta = 0; 
else 
    mu = sqrt(x(1)^2 + sigma);
    if (x(1) <= 0)
        v(1) = x(1) - mu; 
    else
        v(1) = -sigma / (x(1) + mu); 
    end
    beta = 2*(v(1)^2) / (sigma + v(1)^2); 
    v = v/v(1); 
end

