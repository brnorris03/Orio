function [T] = tridiagonalize(A)

% Tridiagonalize a symmetric matrix using Householder transformations. 
% Transformations are performed in place with Householder vectors and coefficients, 
% rather than explicitly forming matrices and explicitly computing their products. 
%
% Returns T = Q^t * A * Q
%
% Source: Golub and Van Loan, "Matrix Computations" p. 415
% 
% Input: 
%     Matrix A        Matrix to tridiagonalize. 
%     
%   Output:
%     Matrix T        Tridiagonal matrix similar to input matrix. 
%     
%     
%   Alex Kaiser, LBNL, 8/2010
  

T = A; 
[m n] = size(T); 
if(m ~= n)
    error('input matrix must be square'); 
end


for k=1:n-2
    [v beta] = householder(T(k+1:n, k)); 
    p = beta * T(k+1:n, k+1:n) * v ;     
    w = p - (beta * p' * v / 2) * v ; 
    T(k+1,k) = norm(T(k+1:n, k)) ; 
    T(k, k+1) = T(k+1, k); 
    T(k+1:n, k+1:n) = T(k+1:n, k+1:n) - v * w' - w * v'; 
end


% zero off diagonal multipliers 
for i = 1:n-2
    for j = i+2:n
        T(i,j) = 0.0 ; 
        T(j,i) = 0.0 ; 
    end
end

    
            
            