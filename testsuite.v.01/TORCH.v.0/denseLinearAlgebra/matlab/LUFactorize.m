function [L U] = LUFactorize( A ) 

%   Computes LU factorization of matrix A. 
%   Returns L and U such that L*U = A
%
%   Source - 
%   http://www.netlib.org/netlib/utk/people/JackDongarra/PAPERS/hpcs-report-11-2007-final.pdf
%
%
%   Input:
%       double A    Matrix to factorize. 
% 
%   Output:
%       double L    Permutation of lower triangular matrix
%       double U    Upper triangular matrix
% 
% 
% 
%   Alex Kaiser, LBNL, 8/2010

  
 
U = A;  %copy 'a' for non-destructive editing

[m n] = size(A); 

if(m ~= n) 
   error('Input matrix must be square. Other dimensions not supported.\n');  
end

pivots = zeros(n,1); 

for k = 1:n-1  %loop over columns 
    
    %partial pivoting
    %get largest element in absolute value from current column to use as pivot    
    [maxFound piv] = max( abs( U(k:n, k))) ;   
    pivots(k) = piv + k-1 ; 
    
    
    if( U(pivots(k), k) == 0.0 ) 
        warning('Hit zero pivot.');  
        continue 
    end
    
    %interchange rows if necessary
    if(k ~= pivots(k))
        temp = U(pivots(k),:);    
        U(pivots(k),:) = U(k,:); 
        U(k,:) = temp ;    
    end
    

    %scale column i below diagonal by 1/A(i,i) 
    for j=k+1:n
        U(j,k) = U(j,k)/U(k,k); 
    end

    % don't actually perform, as all work is done within U array
    % set row i of U
    %    for j=i:n
    %        U(i,j) = A(i,j) ; 
    %    end

    %perform trailing matrix update    
    U(k+1:n, k+1:n) = U(k+1:n, k+1:n) - U(k+1:n,k) * U(k, k+1:n);
    
end     %end of loop over columns

pivots(n) = n; 



% place the result into two seperate arrays, L and U 
L = eye(size(U));  %start with the identity, because L always has ones on diagonal
for j = 1:n-1
    for i = j+1:n
        L(i,j) = U(i,j); 
        U(i,j) = 0.0; 
    end
end


%permute L to make L*U = a
for k=n:-1:1
    if(pivots(k) ~= k)              %interchange if necessary
        temp = L(pivots(k),:)  ;    
        L(pivots(k),:) = L(k,:) ;
        L(k,:) = temp ;
    end
end    

