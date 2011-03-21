function a = getRandomInt(m, n, b)
%
% Returns m*n matrix of random int's in the range [0, 2^b - 1]
%
% Input: 
% int m,n           Dimensions of matrix to be returned. 
% int b             Expondent for max value. 
%
% Output:
% matrix a          m*n matrix of random int's in the range [0, 2^b - 1]
%
%
% Alex Kaiser, LBNL, 9/2010
%

maxVal = 2^b; 
a = floor((maxVal * rand(m,n))) ; 


