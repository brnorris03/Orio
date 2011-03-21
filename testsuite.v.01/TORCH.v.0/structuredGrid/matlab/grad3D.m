function newGrid = grad3D(grid, newGrid, height)

%    Numerically approximates the gradient of the 3D grid in each direction.
% 	 Ghost zones must be included with input.
% 	 Ghost zones are not modified in output arrays.
% 
% 	 Input:
% 	 vector field grid  The vector field of which to approximate divergence.
%                       Ghost zones included.
% 	 grid newGrid       Preallocated, zeroed grid. 
%    double height      Grid height. 
% 
% 	 Output:
% 	 grid newGrid   The divergence of the input grid
%      
%      Alex Kaiser, LBNL, 7/2010
%

     
    %out of place grid update for 3D gradient

    [m n p] = size(grid); 
    
    % pass grid height as parameter
    % height = (n-1)/(2*x)     
    
    alpha = 1 / (2*height); 
    beta  = 1 / (2*height); 
    gamma = 1 / (2*height); 

    % only call function for internal nodes
    % do not update ghost zones on boundaries
    
    for i = 2:m-1
        for j = 2:n-1
            for k = 2:p-1
                newGrid(i,j,k,1) = alpha * (grid(i+1, j, k) - grid(i-1, j, k));
                newGrid(i,j,k,2) = beta  * (grid(i, j+1, k) - grid(i, j-1, k));
                newGrid(i,j,k,3) = gamma * (grid(i, j, k+1) - grid(i, j, k-1));
            end
        end
    end

end





