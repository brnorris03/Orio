function [dx dy dz] = centralDifference3D(grid, dx, dy, dz, height)

%    Numerically approximates the first partial derivatives of the 3D grid g in each direction.
% 	 Ghost zones must be included with input.
% 	 Ghost zones are not modified in output arrays.
% 
% 	 Input:
%    	 grid g  The grid of which to approximate partials. 
%                Ghost zones included. 
%        grid dx  Preallocated, zeroed grid.
%        grid dy  Preallocated, zeroed grid.
%        grid dz  Preallocated, zeroed grid.
%        double height Grid height
% 
% 	 Output:
%        grid dx  Approximation to partial derivative in x direction.
%        grid dy  Approximation to partial derivative in y direction.
%        grid dz  Approximation to partial derivative in z direction.
%
%    Alex Kaiser, LBNL, 7/2010. 


    [m n p] = size(grid); 
    
    % pass grid height as parameter
    % height = (n-1)/(2*x)     
    
    alpha = 1 / (2*height); 
    beta  = 1 / (2*height); 
    gamma = 1 / (2*height); 

    % only call function for internal nodes
    % do not update ghost zones  
    for i = 2:m-1
        for j = 2:n-1
            for k = 2:p-1
                dx(i,j,k) = alpha * (grid(i+1, j, k) - grid(i-1, j, k));
                dy(i,j,k) = beta  * (grid(i, j+1, k) - grid(i, j-1, k));
                dz(i,j,k) = gamma * (grid(i, j, k+1) - grid(i, j, k-1));
            end
        end
    end

end