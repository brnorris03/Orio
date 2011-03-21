function [ rowPtr columnIndices values ] = getSymmetricDiagonallyDominantCSR(n, nnz, distribution) 

% 	 Generates randomized symmetric diagonally-dominant matrices in a banded structure. 
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


    width = floor((n-1)/10) ; 

    % keep a temp array
    rowSum = zeros(n,1); 
    
    
    for band = 0:9
        
        nnzThisBand = floor(distribution(band + 1) * (nnz-n) / 2) ; 

        for j = 1:nnzThisBand
            
            %place lower diagonal element
            nnzSoFar = nnzSoFar + 1; 
            [rowInd colInd] = getRandomIndicesFromBand(n, band, width) ; 
            row(nnzSoFar) = rowInd; 
            column(nnzSoFar) = colInd; 
            currentValue = rand() ; 
            valuesOrig(nnzSoFar) = currentValue ; 
            
            %place above diagonal element for symmetry
            nnzSoFar = nnzSoFar + 1; 
            row(nnzSoFar) = colInd ; 
            column(nnzSoFar) = rowInd ; 
            valuesOrig(nnzSoFar) = currentValue ; 
            
            %adjust the current value placed on the row
            rowSum(rowInd) = rowSum(rowInd) + currentValue ; 
            rowSum(colInd) = rowSum(colInd) + currentValue ;
        end
    
    end
    
    % add diagonal elements
    for j = 1:n
        nnzSoFar = nnzSoFar + 1; 
        row(nnzSoFar) = j; 
        column(nnzSoFar) = j; 
        valuesOrig(nnzSoFar) = 2 * abs(rowSum(j) ) + 1.0 ; 
    end
    
    
    row = row(1:nnzSoFar); 
    column = column(1:nnzSoFar); 
    valuesOrig = valuesOrig(1:nnzSoFar); 
    
    if length( row ) ~= nnzSoFar 
        error('whoa' ); 
    end

    [ rowPtr columnIndices values ] = getCSRfromRowColumn(n, n, row, column, valuesOrig) ; 


end













