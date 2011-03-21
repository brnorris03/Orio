
function coarse = fineToCoarse(coarse, fine)

%    Injection operator from fine grid to coarse grid. 
%
%    If coarse grid is size n in each dimension, fine grid will be of size 2*n-1
%
%    Input:
%     grid coarse      Preallocated 3D coarse array of length (n + 1) / 2 in each dimension
%     grid fine        3D fine array of length n in each dimension (which is computed here)
%     
%     
%     Output:
%     grid coarse      3D array formed using injection operator from fine grid
%                      
%     
%     
%     Alex Kaiser, LBNL, 7/2010


    [n n n] = size(fine) ;  
    
    if( mod(n,2) ~= 1 ) 
        error('We are in a world of resizing pain. Array length must be odd.'); 
    end
    
    coarse = fine(1:2:n, 1:2:n, 1:2:n);
    
end


