function grid = initGrid3D(f, height, x, y, z)

% 	 Initializes, allocates and returns a structured grid according to the
% 	 supplied function.
% 	 
%    Ghost zones included, grid initialized to (m+2, n+2, p+2). 
% 	 Minimum value for each dimension is zero. 
% 
% 	 Input:
%       double f   The function, must be from R3 to reals. f:R^3 -> R
%               
%       double h   Grid height
%       double x   Maximum x value
%       double y   Maximum y value
%       double z   Maximum z value
% 
% 	 Output:
%       grid (returned) Grid structure with values initialized to the function values on mesh points
% 
%    Alex Kaiser, LBNL, 7/2010


m = x/height - 1;
n = y/height - 1;
p = z/height - 1; 
grid = zeros(m+2, n+2, p+2); 
% if this allocation throws a warning, consider it an error
% height must divide x,y,z evenly


for i = 1:m+2
    for j = 1:n+2
        for k = 1:p+2
            grid(i,j,k) = f( (i-1)*height, (j-1)*height, (k-1)*height );
        end
    end
end


 