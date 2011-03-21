function A = symmetricQRStep( A ) 
%
% Single step of a QR factorization. Uses givens rotations. 
% Does not explicitly store information on Q.
%
% Golub and Van Loan, "Matrix Computations," p. 420
%
% Input:
%     Matrix A       Woring matrix. 
%     
%   Output:
%     Matrix A       Modified input matrix.
%
% Alex Kaiser, LBNL, 9/2010
%
    
    [m n] = size(A) ; 

    d = (A(n-1, n-1) - A(n,n)) / 2 ; 

    % if built in matlab sign function is called, Inf will result
    % sign should be 1 or -1. 
    mu = A(n,n) - A(n, n-1)^2 / (d + mySign(d)*sqrt(d^2 + A(n, n-1)^2) ); 
    x = A(1,1) - mu; 
    z = A(2,1); 

    for k = 1:n-1
        A = applyGivens( A, x, z, k, k+1); 
        if (k < n-1)
            x = A(k+1, k); 
            z = A(k+2, k); 
        end
    end

end % end of function 


function s = mySign( x )
% Returns 1 if abs(x) == x
% Use instead of built in sign function, 
%   because that returns sign(0) as zero, which creates problems in this algorithm. 
    if abs(x) == x
        s = 1; 
    else 
        s = -1; 
    end

end


