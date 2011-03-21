function [ rowInd colInd ] = getRandomIndicesFromBand(n, bandNumber, width) 

%   Returns randomized pair in the specified band off the diagonal 
%   Does not ever return i=j, so when using this function to construct a
%       matrix diagonal entries should be handeled manually 
%
%   Band should be indexed from bandNumber * width : bandNumber * width + width
%
%
% 	Input 
% 		int n				Matrix size
% 		int bandNumber		Band of matrix from which to generate a pair. 
% 								Numbered from 0 to 9
% 		int width			Width, in matrix entries, of each band 
% 	 
% 	 Output
% 		int rowInd			Pointer to row index
% 		int colInd			Pointer to column index
% 
%
%   Alex Kaiser, LBNL, 7/2010
%


firstRand = randi(width) ; 
offDiagIndex = bandNumber * width + firstRand  ; 

randVal = randi( 2*n - ( 2 * (bandNumber * width + firstRand) ) )  ; 
diagIndex = bandNumber * width + firstRand + randVal  ; 

rowInd = round( (offDiagIndex + diagIndex) / 2 ) ; 
colInd = round( ( diagIndex - offDiagIndex ) / 2 ) ; 


% make sure parameters are okay with a loop 
% this should hopefully never be called
k = 0 ; 
while rowInd > n || colInd > n || rowInd < 1 || colInd < 1 || rowInd == colInd
    k = k + 1; 
    
    firstRand = randi(width) ; 
    offDiagIndex = bandNumber * width + firstRand  ; 

    randVal = randi( 2*n - ( 2 * (bandNumber * width + firstRand) ) )  ; 
    diagIndex = bandNumber * width + firstRand + randVal  ; 

    rowInd = round( (offDiagIndex + diagIndex) / 2 ) ; 
    colInd = round( ( diagIndex - offDiagIndex ) / 2 ) ; 
    
end

if k > 0 
    'hit problem loop k times' 
    k
end

% check that returned values are in bounds 
if rowInd > n || colInd > n
    rowInd
    colInd
    offDiagIndex
    diagIndex
    error('indices too large!')
end


if rowInd < 1 || colInd < 1
    rowInd
    colInd
    offDiagIndex
    diagIndex
    error('indices too zero or negative!')
end

if rowInd == colInd 
    error('cannot return diagonal entry')
end























