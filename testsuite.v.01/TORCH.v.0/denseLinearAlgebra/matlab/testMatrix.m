function a = testMatrix(n, t)

% Generates a tridiagonal test matrix according to some LAPACK test guidelines. 
% 
% Source: http://www.netlib.org/lapack/lawnspdf/lawn182.pdf
% 
% Input: 
%   int n       Dimension of matrix returned. 
%   int t       Type number. 
%
% Output:
%   matrix a    Symmetric, tridiagonal test matrix. 
%
%
% Alex Kaiser, LBNL, 9/2010



%zero matrix
if t == 0
    a = zeros(n);  

% identity
elseif t == 1
    a = eye(n);

% (1,2,1) tridiagonal
elseif t == 2
    a = zeros(n); 
    for i=1:n-1
        a(i,i) = 2; 
        a(i,i+1) = 1; 
        a(i+1,i) = 1; 
    end
    a(n,n)=2;  

%wilkinson matrix    
elseif t == 3
    a = wilkinson(n); 
    
% Clement matrix 
elseif t == 4
    a = zeros(n); 
    for i=1:n-1 
        a(i+1,i) = sqrt( i * (n+1-i) );  
        a(i,i+1) = sqrt( i * (n+1-i) );
    end
    
%Legendre matrix    
elseif t == 5   
    a = zeros(n); 
    for i=1:n-1
        k = i+1; 
        a(i+1,i) = k / sqrt((2*k-1) * (2*k+1)); 
        a(i,i+1) = k / sqrt((2*k-1) * (2*k+1)); 
    end
    
%Leguerre matrix    
elseif t == 6
    a = zeros(n); 
    for i=1:n-1
        a(i,i) = 2*i + 1; 
        a(i,i+1) = i+1; 
        a(i+1,i) = i+1; 
    end
    a(n,n)=2*n + 1;
    
% Hermite matrix    
elseif t == 7
    a = zeros(n); 
    for i=1:n-1 
        a(i+1,i) = sqrt(i); 
        a(i,i+1) = sqrt(i); 
    end
    
    
else
    error('Unsupported type.'); 
end


  
    
    