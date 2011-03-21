function Q = getRandomOrthogonalMatrix(n)

% Randomly generates an orthogonal matrix Q. 
% Two versions are available:
%     1. Use built in Matlab QR factorization on a random symmetic matrix. 
%        Q is then a randomized orthogonal matrix. Discard R. 
%        
%     2. Perform a series of householder transformations on a random symmetric matrix. 
%        Calculate their product, which is then a randomized orthogonal martix. 
%        This method is preferred for transparency and efficiency and thus default. 
%        This loosely follows the scheme of "The Efficient Generation of Randomized
%        Orthogonal Matrices with an Application to Condition Estimators"
%        R. W. Stewart, SIAM Journal on Numerical Analysis, Vol 17. 
%        
%   
%   Input:
%     int n       Dimension of matrix desired. 
%     
%   Output:
%     matrix Q    Randomized orthoganal matrix. 
%     
%     
%   Alex Kaiser, LBNL, 9/2010


%{
% Use built in QR to generate Q. 
Q = randn(n); 
[Q R] = qr(Q); 
%}


% Perform manually
T = randn(n) ; 
T = T + T';

betaArray = zeros(n-1, 1) ;  
houseHolderVectors = zeros(n); 

for k=1:n-1
    [v beta] = householder(T(k:n, k)); 
    
    betaArray(k) = beta ; 
    houseHolderVectors(k:n,k) = v ; 

end

Q = getQFromHouseholderVectors(houseHolderVectors, betaArray, n-2) ; 

end





function Q = getQFromHouseholderVectors(A, betaArray, q)

% Computes the product of matrices implied by input array of Householder vectors. 
% 
% Input:
%     matrix A            Matrix containing Householder vectors in its columns. 
%     vector betaArray    Array of coefficients for Householder vectors. 
%     int q               Start index. 
%     
% Output:
%     matrix Q            Randomized matrix as product of Householder matrices. 
%    


    Q = eye(size(A)); 
    [n n] = size(A) ; 
    
    
    for j = q:-1:1
       v = A(j:n,j) ; 
       Q(j:n,j:n) = (eye(n-j+1) - betaArray(j) * (v * v')) * Q(j:n,j:n) ;  
    end
      

end
