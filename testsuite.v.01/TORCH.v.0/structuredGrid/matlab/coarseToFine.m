function fine = coarseToFine(coarse, fine)

%    Converts a coarse grid to a fine grid. 
%    Uses an averaging operator
%    If coarse grid is size n in each dimension, fine grid will be of size 2*n-1
%
%   Input:
%     grid coarse  3D coarse array of length n in each dimension (which is computed here)
%     grid fine    Preallocated 3D fine array n of length 2*n - 1 in each dimension
%     
%     Output:
%     grid fine 3D array that is the linear interpolation of the coarse grid
%     
%     
%     Alex Kaiser, LBNL, 7/2010

    
    
    [n n n] = size(coarse); 
    
    fineN = 2*n-1; 
    
    % should be passed allocated. 
    % fine = zeros(2*n-1, 2*n-1, 2*n-1); 
    
    %copy existing points first
    for i=1:n
        for j = 1:n
            for k = 1:n
                fine(2*i - 1, 2*j - 1, 2*k - 1) = coarse(i,j,k); 
            end
        end
    end
        
    % average odd numbered columns in odd numbered planes in x direction
    for i=2:2:fineN-1
        for j = 1:2:fineN
            for k = 1:2:fineN
                fine(i,j,k) = .5 * (fine(i-1,j,k) + fine(i+1,j,k));             
            end
        end
    end
    
    % average even numbered columns in odd numbered planes in y direction
    for i=1:fineN
        for j = 2:2:fineN-1
            for k = 1:2:fineN
                fine(i,j,k) = .5 * (fine(i,j-1,k) + fine(i,j+1,k));             
            end
        end
    end
    
    % average entire even numbered planes in z direction 
    for i=1:fineN
        for j = 1:fineN
            for k = 2:2:fineN-1
                fine(i,j,k) = .5 * (fine(i,j,k-1) + fine(i,j,k+1));             
            end
        end
    end
    
end


