% 
%    Check central difference approximations to divergence in 3D.
% 	 All l2 relative errors and all maximum differences must be under the
% 	 specified tolerance parameters to pass.
%      
% 	 Parameters:
% 	 double x,y,z               Boundaries of grid.
% 	 int n                      Number of grid points to use in x direction.
% 	 double tolNorm             Tolerance for l2 relative error.
% 	 double tolMax              Tolerance for max difference.
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
x = 1; 
y = 1; 
z = 1; 

height = x/(n+1);

tolNorm = 10e-5; 
tolMax = 10e-5; 

toughness = 0.25 ; 

pass = true; 

for type = 0:2
    
    divFn = divergenceTestFn(type, toughness); 

    grid = initVectorField3D( divFn, height, x, y, z) ; 

    %get proper dimensions incl ghost zones as calc in grid initialization
    newGrid = zeros( size(grid) ); 

    fprintf(1,'Results for function of type %d:\n\n', type);
    
    
    % evaluate
    tic; 
    newGrid = div3D(grid, newGrid, height) ;   
    toc; 
    
    
    divSoln = divergenceSoln(type, toughness); 
    
    solnGrid = initGrid3D( divSoln, height, x, y, z) ; 

    % remove ghost zones
    % get dimensions as calc in initGrid
    m = x/height - 1;
    n = y/height - 1;
    p = z/height - 1; 

    solnGrid = solnGrid(2:m+1,2:n+1,2:p+1) ; 
    newGrid = newGrid(2:m+1,2:n+1,2:p+1) ; 

    
    maxDiff = max(max(max(abs(solnGrid - newGrid ))))

    if( maxDiff < tolMax)
        disp('Divergence max diff tests passed.'); 
    else
        disp('Divergence max diff tests failed.'); 
        pass = false ; 
    end

    if( norm(solnGrid(:)) > 0 )
    
        relErr = norm(solnGrid(:) - newGrid(:)) / norm(solnGrid(:)) 

        if( relErr < tolNorm )
            disp('Divergence relative error tests passed.'); 
        else
            disp('Divergence relative error tests failed.'); 
            pass = false; 
        end
    
    else
       	fprintf(1, 'Analytic solution has norm exactly zero, which implies divergence free vector field on type %d.\n', type);
		fprintf(1, 'Relative error test skipped.\n\n');
    end
    
end


if( pass )
    disp('Divergence tests passed.'); 
else
    disp('Divergence tests failed.'); 
end
    
structuredGridPass = structuredGridPass & pass ; 
    