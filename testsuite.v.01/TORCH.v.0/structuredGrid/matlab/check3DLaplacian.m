% 
%    Check central difference approximations to Laplacian in 3D.
% 	 All l2 relative errors and all maximum differences must be under the
% 	 specified tolerance parameters to pass.
%      
% 	 Parameters:
% 	 double x,y,z               Boundaries of grid.
% 	 int n                      Number of grid points to use in x direction.
% 	 double tolNorm             Tolerance for l2 relative error.
% 	 double tolMax              Tolerance for max difference.
%    int type                   Type number for function to return.
%    double toughness           Difficulty parameter. Larger values result in more numerically challenging computations.
% 
% 	 Output:
% 	 Prints whether all norms are under supplied tolerances.
%
%    Alex Kaiser, LBNL, 7/2010
%

disp('Begin Laplacian test.'); 

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

    fprintf(1,'Results for function of type %d:\n\n', type);
    
    %get proper dimensions incl ghost zones as calc in grid initialization
    newGrid = zeros( size(grid) ); 

    tic; 
    newGrid = laplacian3D(grid, newGrid, height) ; 
    toc; 
    
    analyticLaplacian = laplacianSoln(type, toughness); 

    solnGrid = initGrid3D( analyticLaplacian, height, x, y, z) ; 

    % get dimensions as calc in initGrid
    m = x/height - 1;
    n = y/height - 1;
    p = z/height - 1; 

    %remove ghost zones for soln comparison
    solnGrid = solnGrid(2:m+1,2:n+1,2:p+1);  
    newGrid = newGrid(2:m+1,2:n+1,2:p+1) ; 

    
    
    
    relErr = norm(solnGrid(:) - newGrid(:)) / norm(solnGrid(:)) 

    if( relErr < tolNorm )
        disp('Laplacian relative error tests passed.'); 
    else
        disp('Laplacian relative error tests failed.'); 
        pass = false; 
    end


    maxDiff = max(max(max(abs(solnGrid - newGrid ))))

    if( maxDiff < tolMax)
        disp('Laplacian max diff tests passed.'); 
    else
        disp('Laplacian max diff tests failed.'); 
        pass = false; 
    end

end

if( pass )
    disp('Laplacian tests passed.'); 
else
    disp('Laplacian tests failed.'); 
end

structuredGridPass = structuredGridPass & pass ;

