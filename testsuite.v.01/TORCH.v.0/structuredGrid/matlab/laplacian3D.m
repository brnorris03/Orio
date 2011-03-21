function newGrid = laplacian3D(grid, newGrid, height)

%    Numerically approximates the Laplacian of the 3D grid in each direction.
% 	 Ghost zones must be included with input.
% 	 Ghost zones are not modified in output arrays.
% 
% 	 Input:
% 	 grid grid          The grid of which to approximate Laplacian.
%                       Ghost zones included.
% 	 grid newGrid       Preallocated, zeroed grid. 
%    double height      Grid height. 
% 
% 	 Output:
% 	 grid newGrid       The Laplacian of the input grid
%    
%
%    Alex Kaiser, LBNL, 7/2010


    [m n p] = size(grid); 
    
    % height = (n-1)^2 / (x*y) ;   
    alpha = -6 / height^2 ; 
    beta = 1 / height^2; 
    

    % only call function for internal nodes
    % do not update ghost zones on boundaries
    for i = 2:m-1
        for j = 2:n-1
            for k = 2:p-1
                newGrid(i,j,k) = laplacianStencil(grid, i, j, k, alpha, beta); 
            end
        end
    end

end



function stencilValue = laplacianStencil(grid, i, j, k, alpha, beta)
    stencilValue = alpha * grid(i,j,k) +               ...
        beta * (grid(i-1, j, k) + grid(i+1, j, k)) +   ...
        beta * (grid(i, j-1, k) + grid(i, j+1, k)) +   ...
        beta * (grid(i, j, k-1) + grid(i, j, k+1))  ; 
end



