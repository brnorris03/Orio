function grid = initVectorField3D(f, height, x, y, z)

%
%      Initializes, allocates and returns a vector field according to the supplied function.
%      Ghost zones included, grid initialized to (m+2, n+2, p+2).
% 
%      Input:
%      vector f             The function, must be from R3 to R3. f:R^3 -> R^3
%      double height        Grid height
%      double x             Maximum x value
%      double y             Maximum y value
%      double z             Maximum z value
% 
%      Output:
%      grid (returned) vector field with values initialized to the function values on mesh points.
% 
%      Alex Kaiser, LBNL, 7/2010
%


% ghost zones included, grid initialized to m+2 by n+2

% points evaluated from evenly from 0 to x and 0 to y

m = x/height - 1;
n = y/height - 1;
p = z/height - 1; 
grid = zeros(m+2, n+2, p+2, 3); 
% if this allocation throws a warning, consider it an error
% height must divide x,y,z evenly

for i = 1:m+2
    for j = 1:n+2
        for k = 1:p+2
             grid(i,j,k,:) = f( (i-1)*height, (j-1)*height, (k-1)*height );
        end
    end
end

