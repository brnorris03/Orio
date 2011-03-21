function x = gaussSeidelGrid(b, guess, height, dt, maxIt)


% 	 Gauss Seidel method for linear systems.
%    Matrix free version. 
% 	 Specific to the implicit heat equation.
%    Include ghost zones in all grids.
%    Modified from http://www.netlib.org/linalg/html_templates/Templates.html
%    
% 
% 	 Input:
% 	 grid b           Right hand side of the system.
% 	 grid guess       Initial guess, serves to estimate the initial residual.
%    double lambda    dt / h^2, for use with heat eqaution. 
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
tol = 1e-10; 
lambda = dt / (height*height); 

x = guess; 

[m n p] = size(guess); 

r = zeros(m,n,p); 
r = laplacian3D(x,r,height) ; 
r = b - (-dt*r + x)  ; 

iter = 0 ; 

while( norm(r(:)) > tol )
    
    for i = 2:m-1
        for j = 2:n-1
            for k = 2:p-1
                
                sum = (-lambda * (x(i-1,j,k) + x(i+1,j,k) + ...
                                  x(i,j-1,k) + x(i,j+1,k) + ...
                                  x(i,j,k-1) + x(i,j,k+1))) ;  
                
                x(i,j,k) = (b(i,j,k) - sum) / (1 + 6*lambda) ; 
            end
        end
    end

    iter = iter + 1;
    if iter > maxIt
        fprintf(1, 'gs hit max iterations. %d iterations performed.\n', iter ); 
        return ; 
    end

    r = laplacian3D(x,r,height) ; 
    r = b - (-dt*r + x)  ;     
end


fprintf(1, 'gs converged in %d iterations.\n', iter) ; 





