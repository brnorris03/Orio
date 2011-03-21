function A = applyGivens(A, aa, bb, i, k)
% Applies givens rotation to matrix A, computing G'*A*G, and returns the
% rotated matrix. 
% 
%
% Input: 
%   matrix A             Matrix to rotate. 
%   double aa, bb        Specifies the angle of the Givens rotations. 
%   int i, k             Modifies row and column i and k of the input matrix.
% 
% Output:
%   matrix A             Rotated matrix. 
%  
% 
% Alex Kaiser, LBNL, 9/2010
%


    [c s] = myGivens(aa, bb); 

    [m n] = size(A); 

    for j = 1:n
        tau1 = A(i,j); 
        tau2 = A(k,j); 
        A(i,j) = c*tau1 -s*tau2; 
        A(k,j) = s*tau1 + c*tau2;
    end

    for j = 1:m
        tau1 = A(j,i); 
        tau2 = A(j,k); 
        A(j,i) = c*tau1 -s*tau2; 
        A(j,k) = s*tau1 + c*tau2;
    end

end



function [c s] = myGivens(a, b)
% Given a,b, computes c = cos(theta), s = sin(theta) s.t.
% [ c s; -s c]' * [a;b] = [r 0]
%
%
% Input: 
%    double a, b        Specifies the angle of the Givens rotations. 
%
% Output:
%    double [c s]       Values of trigonometric functions for performing rotations. 
%                        
%


    if(b == 0)
        c = 1; 
        s = 0; 
    else
        if(abs(b) > abs(a))
            tau = -a/b; 
            s = 1/sqrt(1+tau^2); 
            c = s*tau; 
        else
            tau = -b/a; 
            c = 1/sqrt(1+tau^2); 
            s = c*tau;
        end
    end

end
