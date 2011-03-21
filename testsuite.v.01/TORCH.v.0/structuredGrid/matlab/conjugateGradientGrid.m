function x = conjugateGradientGrid(b, guess, height, dt, maxIt) 

% 
% 	 Conjugate gradient method for linear systems.
%    Matrix free version. 
% 	 Specific to the implicit heat equation.
%    Include ghost zones in all grids.
%    Modified from 'Matrix Computations' Golub and Van Loan
% 
% 	 Input:
% 	 grid b           Right hand side of the system.
% 	 grid guess       Initial guess.
%    double height    Grid height. 
% 	 double dt        Time step.
% 	 double maxIt     Maximum number of iterations to perform.
% 
% 	 Output:
% 	 grid x (returned) The solution.
%      
% 
%    Alex Kaiser, LBNL, 7/2010



% initial values and tolerance parameters
neps = eps(); 
x = guess; 

%r = b - spmv(x, n, n, rowPtr, columnIndices, values); 
r = zeros(size(guess)); 
r = laplacian3D(x,r,height) ; 
r = b - (-dt*r + x)  ; 

rho = norm(r(:))^2 ;   % L_2 norm squared

% allocating 
w = zeros(size(guess)); 


iter = 0 ; 
while( sqrt(rho) > neps * norm(b(:)) )
    
    if( iter == 0)
        p = r; 
    else
        beta = rho / lastRho;  
        p = r + beta * p; 
    end

    %w = spmv(p, n, n, rowPtr, columnIndices, values); 
    w = laplacian3D(p,w,height) ; 
    w = -dt * w + p ;  
    alpha = rho / (p(:)' * w(:));
    x = x + alpha * p; 
    r = r - alpha * w; 
    lastRho = rho; 
    rho = norm(r(:))^2; 
            
    iter = iter + 1;
    if iter > maxIt
        fprintf(1, 'cg grid hit max iterations. %d iterations performed.\n', iter );
        return ; 
    end
    
end

fprintf(1, 'cg grid converged in %d iterations.\n', iter) ; 

