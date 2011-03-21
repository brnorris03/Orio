% 
% 	 Verification for GMRES based linear system solve.
% 		- Generates general matrix with banded structure according to parameters
% 		- Randomly generates solution to linear system
% 		- Uses matrix multiply to generate right hand side
% 		- Solves system and compares for errors with L2 norm
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
%


if( ~(exist('sparseLinearAlgebraPass', 'var')))
    sparseLinearAlgebraPass = true ; 
end


n = 101; 
nnz = ceil(0.5 * n * n) ; 
distribution = .01 * [65.8 11.4 5.84 6.84 2.85 1.86 1.44 2.71 .774 .387] ; 
tol = 1e-10; 
algorithmTol = .01 * tol; 

maxIt = n; 

disp('allocation time:'); 
tic 
[ rowPtr columnIndices values ] = getGeneralBandedCSR(n, nnz, distribution); 
%getSymmetricDiagonallyDominantCSR(n, nnz, distribution) ; 
toc

%select a solution first, matrix multiply to get values
solution = rand(n,1) ; 
b = spmv(solution, n, n, rowPtr, columnIndices, values);

% set guess and preconditioner matrix
x = zeros(n,1);

% no idea what's optimal here
%d = ones(n,1) + rand(n,1) ; 
%M = diag(d); 
M = eye(n); 
restrt = 20;

%big
max_it = n; 

disp('gmres with spmv time:'); 
tic
[x, error, iter, flag] = gmres(rowPtr, columnIndices, values, n, x, b, M, restrt, max_it, algorithmTol); 
toc

%{
disp('matlab internal time using dense matrix and backslash:'); 
dense = toDense(n, n, rowPtr, columnIndices, values);
tic
builtIn = dense \ b ; 
toc

disp('max diff with built in:'); 
maxDiff = max(abs( builtIn - x))
%}


disp('compare with orininal solution:'); 
maxDiff = max(abs(x - solution))

if maxDiff < tol
    disp('Max diff test passed.');
    sparseLinearAlgebraPass = sparseLinearAlgebraPass & true ;
else
    disp('Max diff test failed.'); 
    sparseLinearAlgebraPass = false ; 
end

disp('relative error with original solution:'); 
if norm(solution) == 0.0
    if norm(x) == 0.0
        relErr = 0.0
    else
        relErr = Inf 
    end
else
    relErr = norm(x - solution) / norm(solution) 
end

if relErr < tol
    disp('Relative error test passed.'); 
    sparseLinearAlgebraPass = sparseLinearAlgebraPass & true ;
else
    disp('Relative error test failed.');
    sparseLinearAlgebraPass = false ;
end
