% 
%    Check central difference approximations to curl in 3D.
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



disp('Begin Curl tests.'); 


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


toughness = 0.25 ; 

pass = true; 

for type = 0:2
    
    curlFn = curlTestFn(type, toughness); 

    grid = initVectorField3D(curlFn, height, x, y, z) ; 

    %get proper dimensions incl ghost zones as calc in grid initialization
    newGrid = zeros( size(grid) ); 

    fprintf(1,'Results for function of type %d:\n\n', type);
    
    tic; 
    newGrid = curl3D(grid, newGrid, height) ;    
    toc; 
    
    analyticCurl = curlSoln(type, toughness); 
    
    solnGrid = initVectorField3D( analyticCurl, height, x, y, z) ; 

    % remove ghost zones
    % get dimensions as calc in initGrid
    m = x/height - 1;
    n = y/height - 1;
    p = z/height - 1; 

    solnGrid = solnGrid(2:m+1,2:n+1,2:p+1,:) ; 
    newGrid = newGrid(2:m+1,2:n+1,2:p+1,:) ; 


    tempNew = newGrid(:,:,:,1) ; 
    tempSoln = solnGrid(:,:,:,1) ; 
    relErrX = norm(tempNew(:) - tempSoln(:)) / norm(tempSoln(:)) ; 
    maxDiffX = max(abs(tempNew(:) - tempSoln(:))) ; 

    tempNew = newGrid(:,:,:,2) ; 
    tempSoln = solnGrid(:,:,:,2) ; 
    relErrY = norm(tempNew(:) - tempSoln(:)) / norm(tempSoln(:)) ; 
    maxDiffY = max(abs(tempNew(:) - tempSoln(:))) ; 

    tempNew = newGrid(:,:,:,3) ; 
    tempSoln = solnGrid(:,:,:,3) ; 
    relErrZ = norm(tempNew(:) - tempSoln(:)) / norm(tempSoln(:)) ; 
    maxDiffZ = max(abs(tempNew(:) - tempSoln(:))) ; 

    maxDiffX
    maxDiffY 
    maxDiffZ 

    if( (maxDiffX < tolMax) && (maxDiffY < tolMax) && (maxDiffZ < tolMax))
        disp('Curl max diff tests passed.'); 
    else
        disp('Curl max diff tests failed.'); 
        pass = false ; 
    end
    
    if (norm(solnGrid(:,1)) > 0) || (norm(solnGrid(:,2)) > 0) || (norm(solnGrid(:,3)) > 0)  
        
        relErrX
        relErrY
        relErrZ
        
        if( (relErrX < tolNorm) && (relErrY < tolNorm) && (relErrZ < tolNorm))
            disp('Curl relative error tests passed.'); 
        else
            disp('Curl relative error tests failed.');
            pass = false ;
        end

    else
        fprintf(1,'Analytic solution is exactly zero, implying a conservative vector field on type %d.\n', type);
        fprintf(1,'Relative error test skipped.\n\n');
    end


end


if( pass )
    disp('Curl tests passed.'); 
else
    disp('Curl tests failed.'); 
end

structuredGridPass = structuredGridPass & pass ; 

