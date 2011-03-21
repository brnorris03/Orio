function x = conjugateGradient(b, n, rowPtr, columnIndices, values, guess, maxIt) 

% 	 Conjugate gradient solve. 
% 	 Solves the linear system A * x = b. 
%    Source - 'Matrix Computations' Golub and Van Loan
% 	 
% 	 Input:
% 		double b				RHS of system. 
%       int n                   Size of system. 
%       int rowPtr              Row array of CSR matrix. 
%       int columnIndices       Column array of CSR matrix. 
%       double values           Values of non-zero matrix entries. 
% 		double guess			Initial guess for the solution of the system.  
% 		int maxIt				Maximum number of iterations to perform. 
% 									Algorithm is guaranteed to find a solution (if it exists)
% 									if this parameter is the dimension of the system or larger.
% 	 
% 	 Output:
% 		double x            	Solution to the linear system
%
%    Alex Kaiser, LBNL, 7/2010
% 


%initial values and tolerance parameters 
neps = eps() ; 
x = guess; 

%compute residual 
r = b - spmv(x, n, n, rowPtr, columnIndices, values); 
rho = norm(r)^2 ;   % L_2 norm squared

iter = 0 ; 
while( sqrt(rho) > neps * norm(b) )

    if( iter == 0 )
        p = r; 
    else
        beta = rho / lastRho;  
        p = r + beta * p; 
    end

    w = spmv(p, n, n, rowPtr, columnIndices, values); 
    alpha = rho / (p' * w); 
    x = x + alpha * p; 
    r = r - alpha * w; 
    lastRho = rho; 
    rho = norm(r)^2; 
    
    iter = iter + 1;
    if iter > maxIt
        fprintf(1, 'cg hit max iterations. %d iterations performed.\n', iter ); 
        return ; 
    end
end

fprintf(1, 'cg converged in %d iterations.\n', iter) ; 







