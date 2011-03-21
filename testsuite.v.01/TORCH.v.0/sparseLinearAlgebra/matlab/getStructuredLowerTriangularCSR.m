function [ rowPtr columnIndices values ] = getStructuredLowerTriangularCSR(n, nnz, distribution) 

% 	 Generates randomized lower triangular matrices in a banded structure.
% 	 Roughly follows the scheme of 
% 	 
% 	 "Benchmarking Sparse Matrix-Vector Multiply in Five Minutes"
% 	 Hormozd Gahvari, Mark Hoemmen, James Demmel, and Katherine Yelick 
% 	 UC Berkeley.
% 
%   Input:
% 		int n						Size of matrix, select such that (n mod(10) == 1) 
%                                       to ensure that corner of matrix isn't left unfilled
% 		int nnz						Approximate number of nonzeros. Actual number may be slightly 
%                                       lower because of possibility of duplicate entries
% 										which are removed in generator.
% 		double distribution[10]		Matrix is divided into ten bands, each including approximately
%                                       distribution[i]*nnz nonzero entires
%
%   Output:
%       int rowPtr              Row pointer vector
%       int columnIndices       Column indices vector
%       double values           Values of matrix entries
%         
%         
%      Alex Kaiser, LBNL, 7/2010
% 


    if nnz < n
        error('must store at least n non-zeros for this routine, as n non zeros are placed on the diagonal'); 
    end

    if length(distribution) ~= 10
        error('distribution must have ten regions') ; 
    end

    if sum(distribution) > 1
       error('distribution must sum to less than one') ;  
    end

    row = zeros(nnz,1); 
    column = zeros(nnz,1); 
    valuesOrig = zeros(nnz,1); 

    nnzSoFar = 0; 
    % add diagonal elements

    for j = 1:n
        nnzSoFar = nnzSoFar + 1; 
        row(nnzSoFar) = j; 
        column(nnzSoFar) = j; 
        valuesOrig(nnzSoFar) = 10.0 ; % 1 + 10 * rand() ; 
    end

    width = floor((n-1)/10) ; 

    
    for band = 0:9
        
        nnzThisBand = floor(distribution(band + 1) * (nnz-n) ) ; 

        for j = 1:nnzThisBand
            nnzSoFar = nnzSoFar + 1; 
            [rowInd colInd] = getRandomIndicesFromBand(n, band, width) ; 
            row(nnzSoFar) = rowInd; 
            column(nnzSoFar) = colInd; 
            valuesOrig(nnzSoFar) = rand() ; 
        end
    
    end
    
    row = row(1:nnzSoFar); 
    column = column(1:nnzSoFar); 
    valuesOrig = valuesOrig(1:nnzSoFar); 
    
    if length( row ) ~= nnzSoFar 
        error('Vector dimensions must agree to construct CSR matrix properly.' ); 
    end

    [ rowPtr columnIndices values ] = getCSRfromRowColumn(n, n, row, column, valuesOrig) ; 


end













