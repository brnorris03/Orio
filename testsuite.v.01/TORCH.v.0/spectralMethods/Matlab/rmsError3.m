function err = rmsError3(x, y)
%
% Calculates the rms error between 3d arrays x and y. 
%
% Input:
% 3D array x        First 3D array to check. 
% 3D array y        Second 3D array to check.
% 
% Output
% err             RMS error between two input 3D arrays. 
% 
% 
% Alex Kaiser, LBNL, 9/2010
%

if size(x) ~= size(y)
    error('input must be of equal size')
end

[m n p] = size(x); 

err = 0; 
for j = 1:m
    for k = 1:n
        for l = 1:p
            err = err + (x(j,k,l) - y(j,k,l))^2; 
        end
    end
end

err = sqrt(err / (m*n*p) ); 