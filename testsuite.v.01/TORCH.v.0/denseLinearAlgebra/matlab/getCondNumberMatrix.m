function a = getCondNumberMatrix(n, cond, type)
%
% Returns randomized matrix following a simplified version of the 
% LAPACK generator function 'DLATMR'
% 
% Off diagonal entries are random from built in 'rand' 
% Diagonal entries are specified from the parameters 'cond' and 'type' 
% 
% 
% Source:
% LAPACK Working Note 9
% A Test Matrix Generation Suite
% James Demmel, Alan McKenney
% http://www.netlib.org/lapack/lawnspdf/lawn09.pdf
% 
% Input:
% int n                 Matrix dimension. 
% double cond           Parameter to influence condition number. 
% int type              Specifies the distribution of entries on the matrix diagonal.  
%                         Suppose D is an array of diagonal entries. 
% 
%                         Type 1: 
%                             D(1) = 1
%                             D(2:n) = 1 / cond
% 
%                         Type 2: 
%                             D(1:n-1) = 1
%                             D(n) = 1 / cond
% 
%                         Type 3:
%                             D(i) form a geometric sequence from 1 to 1/cond
% 
%                         Type 4:
%                             D(i) form an arithmetic sequence from 1 to 1/cond
% 
%                         Type 5: 
%                             D(i) are random from the same distribution 
%                                 as the off diagonal entries. 
%                             This is default if type 1-4 are not selected. 
%                         
% 
% Output:
% matrix a (returned)   Randomized matrix with specified diagonal entries. 
%                         
% Alex Kaiser, LBNL, 9/2010
%                     



a = rand(n) ; 

if type == 1
    a(1,1) = 1 ; 
    for i = 2:n
        a(i,i) = 1 / cond ; 
    end

elseif type == 2
    for i = 1:n-1
        a(i,i) = 1.0 ; 
    end
    a(n,n) = 1/cond ; 
     
elseif type == 3
    coeff = nthroot(1/cond, n-1) ; 
    a(1,1) = 1.0 ; 
    for i = 2:n
        a(i,i) = coeff * a(i-1,i-1) ; 
    end
    
elseif type == 4
    coeff = -(1 - (1/cond)) / (n-1); 
    a(1,1) = 1.0 ; 
    for i = 2:n
        a(i,i) = coeff + a(i-1,i-1) ; 
    end
    
end



