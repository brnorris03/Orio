function D = symmetricQR( A ) 

% Performs symmetric QR factorization on input matrix  
% to compute eigenvalues of the matrix. Q is not explicitly formed. 
% 
% Source - Golub and Van Loan, Matrix Computations, p. 421. 
% 
% Input:
%   Matrix A  Square, symmetric matrix of which to compute eigenvalues. 
% 
% Output:
%   Matrix D  Diagonal matrix, non-zeros of which are the eigenvalues of A
% 
% 
% Alex Kaiser, LBNL, 9/2010



D = tridiagonalize( A ) ;  

q = 0;  

[m n] = size(A); 
tol = 10^-16; %use machine epsilon at 1 for the starting tolerance. 

maxIt = 1000000; 
iteration = 0; 


while(q < n-1)
    
    for i = 1:n-1
       if ( abs( D(i+1,i) ) <= tol * (abs( D(i,i) ) + abs( D(i+1,i+1) )) ) 
           D(i+1, i) = 0; 
           D(i, i+1) = 0; 
       end
    end
    
    
    % find largest diagonal block in lower right of matrix
    start = n-q; %loop index starts with previously found diagonal block
    for k = start:-1:1
        
        %check off-diagonal entries
        %only above diagonal is checked because of symmetry
        if (k ~= 1)
            if(D(k,k-1) == 0)
                q = q+1; 
            else 
                break
            end
        end
    end
    
    
    p = n-q-1 ;
    for k = n-q:-1:2
        if (D(k,k-1) ~= 0) 
            p = p-1 ; 
        else 
            break 
        end 
    end

    
    if (q < n-1)
        D(p+1:n-q, p+1:n-q) = symmetricQRStep( D(p+1:n-q, p+1:n-q) ) ; 
    end       
    
    iteration = iteration+1; 
    if (iteration > maxIt)
        disp('maxIt iterations reached. Returning.'); 
        return ; 
    end
    
end

