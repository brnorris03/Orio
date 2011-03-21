%
% See Structured Grid directory for more rigorous and expanded
%	verification of this kernel. 
%
% Checks sparse matrix vector multiplication for basic correctness.
% Generates random sparse matrix and multiplies, then compares 
%   results to dense matrix version of same.  
% Not a full or true verification scheme because verification is more 
%   computationally intensive than kernel in question, but usefull 
%   for basic correctness of computation. 
%   
% 	 Parameters:
% 
% 		int n						Size of matrix, select such that (n mod(10) == 1) 
%                                       to ensure that corner of matrix isn't left unfilled
% 		int nnz						Approximate number of nonzeros. Actual number may be slightly 
%                                       lower because of possibility of duplicate entries
% 										which are removed in generator.
% 		double distribution[10]		Matrix is divided into ten bands, each including approximately
%                                       distribution[i]*nnz nonzero entires
% 		double tol					Relative error must be less that this value or test will return failure
% 
%
%   Alex Kaiser, LBNL, 7/2010
%


n = 1001; 
avgBandwidth = 10; 
nnz = ceil(0.25 * n * n) ; 
distribution = .01 * [65.8 11.4 5.84 6.84 2.85 1.86 1.44 2.71 .774 .387] ; 

tol = 1e-10; 

disp('dimensions = '); 
n
disp('allocation time:'); 
tic
[ rowPtr columnIndices values ] = getSymmetricDiagonallyDominantCSR(n, nnz, distribution); 
toc


x = randn(n,1);

dense = toDense(n, n, rowPtr, columnIndices, values ) ; 

disp('spmv time:'); 
tic
y = spmv(x, n, n, rowPtr, columnIndices, values); 
toc


builtIn = dense*x ; 


disp('max diff with built in:'); 
maxDiff = max(abs(builtIn - y))
 

disp('relative error compared to dense multiply:'); 
if norm(builtIn) == 0.0
    if norm(y) == 0.0
        relErr = 0.0
    else 
        relErr = Inf
    end
else
    relErr = norm(y - builtIn) / norm(builtIn) 
end

if relErr < tol
    disp('Test passed'); 
else
    disp('Test failed'); 
end
    






