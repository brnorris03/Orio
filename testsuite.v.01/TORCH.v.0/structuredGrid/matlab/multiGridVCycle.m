function x = multiGridVCycle(b, n, lambda, height, dt, guess, depth, maxItDown, maxItUp) 

% 
% 	 Multigrid method for linear systems.
%    Matrix free. 
% 	 Specific to the implicit heat equation.
%    Include ghost zones in all grids.
%    Modified from http://www.netlib.org/linalg/html_templates/Templates.html
%    
% 
% 	 Input:
% 	 grid b             Right hand side of the system.
%    int n              Size of the 3D array in question. 
%    double lambda      dt / h^2, for use with heat eqaution. 
%    double height      Grid height. 
% 	 double dt          Time step.
% 	 grid guess         Initial guess, serves to estimate the initial residual.
%    int depth          Number of steps to descend in V-cycle. 
% 	 int maxItDown      Maximum number of iterations to perform on down cycle.
% 	 int maxItUp        Maximum number of iterations to perform on up cycle.
% 	 
% 	 Output:
% 	 grid x (returned) The solution.
%      
% 
%    Alex Kaiser, LBNL, 7/2010



if(2^depth >= n)
    error('depth must be shallow enough that grid is not divided past one element'); 
end

%allocate grids
grids(depth).guess = 0; 
grids(depth).rhs = 0;
grids(depth).lambda = 0 ; 
grids(depth).height = 0 ; 
grids(depth).length3DArray = 0; 


% initialize most fine grid
grids(1).guess = guess; 
grids(1).rhs = b; 
grids(1).lambda = lambda ; 
grids(1).height = height ; 
grids(1).length3DArray = n + 2; % add two for boundary points 

for j = 2:depth
    grids(j-1).guess = gaussSeidelGrid(grids(j-1).rhs, grids(j-1).guess, grids(j-1).height, dt, maxItDown); 
    
    %allocate and restrict to next coarser grid
    grids(j).length3DArray = ((grids(j-1).length3DArray + 1) / 2) ; 
    
    grids(j).rhs = zeros(grids(j).length3DArray, grids(j).length3DArray, grids(j).length3DArray); 
    
    tempResidual = zeros(size(grids(j-1).guess)) ; 
    tempResidual = laplacian3D(grids(j-1).guess, tempResidual, grids(j-1).height) ; 
    tempResidual = grids(j-1).guess - (-dt * tempResidual + grids(j-1).guess) ; 
    
    grids(j).rhs = fineToCoarse(grids(j).rhs, tempResidual) ; 
     
    grids(j).guess = zeros(grids(j).length3DArray, grids(j).length3DArray, grids(j).length3DArray);    
    grids(j).guess = fineToCoarse(grids(j).guess, grids(j-1).guess) ; 
        
    grids(j).lambda = 0.25 * grids(j-1).lambda ; 
    grids(j).height = 2 * grids(j-1).height; 
    
end

% full solve at coarsest level, allow for maximum iterations
disp('full solve at coarsest level:');  
grids(depth).guess = gaussSeidelGrid(grids(depth).rhs, grids(depth).guess, grids(depth).height, dt, grids(depth).length3DArray^3 );  


for j = depth:-1:2 
    % note: 
    % grids(j) = coarse, grids(j-1) = fine
    
    %interpolate to fine grid
    grids(j-1).guess = grids(j-1).guess + coarseToFine(grids(j).guess, grids(j-1).guess) ; 
    
    % relax on fine grid
    grids(j-1).guess = gaussSeidelGrid(grids(j-1).rhs, grids(j-1).guess, grids(j-1).height, dt, maxItUp); 
        
end

x = grids(1).guess ; 

end











