function newGrid = curl3D(grid, newGrid, height)

% 
% 	 Numerically approximates the curl of the 3D vector field v.
% 	 Ghost zones must be included with input.
% 	 Ghost zones are not modified in output arrays.
% 
% 	 Input:
% 	 vector field grid          The vector field of which to approximate divergence. Ghost zones included.
% 	 vector field newGrid       Address of preallocated, zeroed vector field.
%    double height              Grid height. 
% 
% 	 Output:
% 	 vectorField *curl   Curl of the supplied grid.
%      
%      Alex Kaiser, LBNL, 7/2010
%


    [m n p junk] = size(grid); 
    

    % height = (n-1)/(2*x)         
    alpha = 1 / (2*height); 
    beta  = 1 / (2*height); 
    gamma = 1 / (2*height); 

    % only call function for internal nodes
    % do not update ghost zones on boundaries
    for i = 2:m-1
        for j = 2:n-1
            for k = 2:p-1
                newGrid(i,j,k,1) = beta  * (grid(i, j+1, k, 3) - grid(i, j-1, k, 3)) - ...
                    gamma * (grid(i, j, k+1, 2) - grid(i, j, k-1, 2)) ;
                
                newGrid(i,j,k,2) = gamma * (grid(i, j, k+1, 1) - grid(i, j, k-1, 1)) - ...
                    alpha * (grid(i+1, j, k, 3) - grid(i-1, j, k, 3)) ; 
                
                newGrid(i,j,k,3) = alpha * (grid(i+1, j, k, 2) - grid(i-1, j, k, 2)) - ...
                    beta  * (grid(i, j+1, k, 1) - grid(i, j-1, k, 1)) ;
            end
        end
    end

end

