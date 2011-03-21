
%
% 	 Check central difference approximations to partial derivatives in 3D.
% 	 All l2 relative errors and all maximum differences must be under the
% 	 specified tolerance parameters to pass.
%
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


disp('Begin Central differences tests.'); 


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

toughness = 0.5  

pass = true; 

for type = 0:2
    
    f = centralDiffsTestFn(type, toughness); 

    grid = initGrid3D(f, height, x, y, z) ;

    dx = zeros( size(grid) );
    dy = zeros( size(grid) );
    dz = zeros( size(grid) );

    fprintf(1,'Results for function of type %d:\n\n', type); 
    tic; 
    [dx dy dz] = centralDifference3D(grid, dx, dy, dz, height) ;   
    toc; 
 
    dxFn = centralDiffsDxSoln(type, toughness);
    dyFn = centralDiffsDySoln(type, toughness);
    dzFn = centralDiffsDzSoln(type, toughness);
    

    solnGridDx = initGrid3D( dxFn, height, x, y, z) ; 
    solnGridDy = initGrid3D( dyFn, height, x, y, z) ; 
    solnGridDz = initGrid3D( dzFn, height, x, y, z) ; 


    % remove ghost zones
    % get dimensions as calc in initGrid
    m = x/height - 1;
    n = y/height - 1;
    p = z/height - 1; 

    solnGridDx = solnGridDx(2:m+1,2:n+1,2:p+1) ;
    solnGridDy = solnGridDy(2:m+1,2:n+1,2:p+1) ;
    solnGridDz = solnGridDz(2:m+1,2:n+1,2:p+1) ;

    dx = dx(2:m+1,2:n+1,2:p+1) ; 
    dy = dy(2:m+1,2:n+1,2:p+1) ; 
    dz = dz(2:m+1,2:n+1,2:p+1) ; 
    
    
    relErrDx = norm(solnGridDx(:) - dx(:)) / norm(solnGridDx(:)) 
    relErrDy = norm(solnGridDy(:) - dy(:)) / norm(solnGridDy(:)) 
    relErrDz = norm(solnGridDz(:) - dz(:)) / norm(solnGridDz(:))

    
    if( (relErrDx < tolNorm) && (relErrDy < tolNorm) && (relErrDz < tolNorm))
        disp('Central difference relative error tests passed.'); 
    else
        disp('Central difference relative error tests failed.'); 
        pass = false; 
    end

    maxDiffDx = max(max(max(abs(solnGridDx - dx ))))
    maxDiffDy = max(max(max(abs(solnGridDy - dy ))))
    maxDiffDz = max(max(max(abs(solnGridDz - dz ))))

    if( (maxDiffDx < tolMax) && (maxDiffDy < tolMax) && (maxDiffDz < tolMax))
        disp('Central difference max diff tests passed.'); 
    else
        disp('Central difference max diff tests failed.'); 
        pass = false; 
    end

end



if( pass )
    disp('All central difference tests passed.'); 
else
    disp('Central difference tests failed.'); 
end

structuredGridPass = structuredGridPass & pass ; 
