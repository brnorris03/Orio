function err = rmsError(x, y)
%
% Calculates the rms error between vectors x and y. 
%
% Input:
% vector x        First vector to check. 
% vector y        Second vector to check.
% 
% Output
% err             RMS error between two input vectors. 
% 
% 
% Alex Kaiser, LBNL, 9/2010
%


if (~(isvector(x) && (isvector(y))) || (length(x) ~= length(y)) )
    error('input must be vectors of equal length')
end

err = 0; 
for k = 1:length(x)
    err = err + (x(k) - y(k))^2;  
end

err = sqrt(err / length(x)); 




