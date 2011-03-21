
% 
%    Check central difference approximations to gradient in 3D.
% 	 All l2 relative errors and all maximum differences must be under the
% 	 specified tolerance parameters to pass.
%      
% 	 Parameters:
% 	 double x,y,z               Boundaries of grid.
% 	 int n                      Number of grid points to use in x direction.
% 	 double tolNorm             Tolerance for l2 relative error.
% 	 double tolMax              Tolerance for max difference.
%    int type   Type number for function to return.
%    double toughness  Difficulty parameter. Larger values result in more numerically challenging computations.
% 
% 	 Output:
% 	 Prints whether all norms are under supplied tolerances.
%
%    Alex Kaiser, LBNL, 7/2010
%


disp('Begin Divergence tests.');


if( ~(exist('structuredGridPass', 'var')))
    structuredGridPass = true ; 
end

% grid height determined by n and x exclusively
% other dimensions will be expanded to grid is evenly spaced
n = 100; 
x = 1.0; 
y = 1.0; 
z = 1.0; 

height = x/(n+1);

tolNorm = 10e-5; 
tolMax = 10e-5;

toughness = 0.5 ; 

pass = true; 

for type = 0:2
    
    f = centralDiffsTestFn(type, toughness); 

    grid = initGrid3D(f, height, x, y, z) ;

    %get proper dimensions incl ghost zones as calc in grid initialization
    newGrid = zeros( [size(grid) 3] );  

    fprintf(1,'Results for function of type %d:\n\n', type);
    
    tic; 
    newGrid = grad3D(grid, newGrid, height) ;   
    toc; 
    
    
    gradFn = gradientSoln(type, toughness) ; 
    
    solnGrid = initVectorField3D( gradFn, height, x, y, z) ; 

    %remove ghost zones
    % get dimensions as calc in initGrid
    m = x/height - 1;
    n = y/height - 1;
    p = z/height - 1; 

    solnGrid = solnGrid(2:m+1,2:n+1,2:p+1,:) ; 
    newGrid = newGrid(2:m+1,2:n+1,2:p+1,:) ; 

    
    tempNew = newGrid(:,:,:,1) ; 
    tempSoln = solnGrid(:,:,:,1) ; 
    relErrX = norm(tempNew(:) - tempSoln(:)) / norm(tempSoln(:)) 
    maxDiffX = max(abs(tempNew(:) - tempSoln(:))) ; 

    tempNew = newGrid(:,:,:,2) ; 
    tempSoln = solnGrid(:,:,:,2) ; 
    relErrY = norm(tempNew(:) - tempSoln(:)) / norm(tempSoln(:)) 
    maxDiffY = max(abs(tempNew(:) - tempSoln(:))) ; 

    tempNew = newGrid(:,:,:,3) ; 
    tempSoln = solnGrid(:,:,:,3) ; 
    relErrZ = norm(tempNew(:) - tempSoln(:)) / norm(tempSoln(:)) 
    maxDiffZ = max(abs(tempNew(:) - tempSoln(:))) ; 


    if( (relErrX < tolNorm) && (relErrY < tolNorm) && (relErrZ < tolNorm))
        disp('Gradient relative error tests passed.'); 
    else
        disp('Gradient relative error tests failed.'); 
        pass = false; 
    end

    maxDiffX
    maxDiffY 
    maxDiffZ 

    if( (maxDiffX < tolMax) && (maxDiffY < tolMax) && (maxDiffZ < tolMax))
        disp('Gradient max diff tests passed.'); 
    else
        disp('Gradient max diff tests failed.'); 
        pass = false; 
    end

end

if( pass )
    disp('Gradient tests passed.'); 
else
    disp('Gradient tests failed.'); 
end

structuredGridPass = structuredGridPass & pass ; 


