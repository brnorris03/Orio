function [ rowPtr columnIndices values ] = getCSRfromRowColumn(m, n, row, column, valuesOrig )

%  	 Generates a CSR matrix from a matrix in (row, col, val) format. 
% 	 Values are sorted and duplicates are removed, so the (row, col, val) arrays
% 		can come in any order. 
% 	 
% 	 Input:
% 		int m					Number of rows. 
% 		int n					Number of columns. 
% 		int row[]				Row indices. 
% 		int column[]			Column indices. 
% 		double valuesOrig[]		Values of matrix entries. 
% 		int origLength			Original length of array, which may contain duplicate entries. 
% 	 
% 	 Output:
%                                 Original matrix in CSR format. 
%       int rowPtr              Row pointer vector. 
%       int columnIndices       Column indices vector. 
%       double values           Values of matrix entries.    
%   
%
%      Alex Kaiser, LBNL, 7/2010
%




% need to handle the problem that matrices with a row of all zeros are not handled
% properly, check this. 


    % if the input arrays are not sorted, sort them
    if( ~isSortedAndNoDuplicates(row, column, valuesOrig) ) 

        [row column valuesOrig] = sortTriples(row, column, valuesOrig, 1, length(row) ) ;  
        
        
        % remove duplicate elements 
        % allocate for 20% repeats, more will cause resizing
        sizeAllocated = floor(.2 * length(row) ) ; 
        toRemove = zeros( sizeAllocated, 1 ) ; 
        numToRemove = 0; 
        for j = 2:length(row)

            if ((row(j) == row(j-1)) && (column(j) == column(j-1))) 
                numToRemove = numToRemove + 1; 
                toRemove(numToRemove) = j; 
            end

            if numToRemove > sizeAllocated
                toRemove = [ toRemove zeros(size(toRemove)) ] ; 
                sizeAllocated = 2 * sizeAllocated ;
            end


        end

        if numToRemove > 0 
            row( toRemove(1:numToRemove) ) = [] ; 
            column( toRemove(1:numToRemove) ) = [] ; 
            valuesOrig( toRemove(1:numToRemove) ) = [] ; 
        end
    end 
    
    % if they are sorted and without duplicates, carry on
    rowPtr = zeros(1, m+1) ; 
    columnIndices = column ; 
    values = valuesOrig ; 
   
    rowPtr(1) = 1 ;
    rowIndex = 1 ; 
    
    for j = 1:length(row)
        if row(j) > rowIndex
            rowIndex = rowIndex + 1; 
            rowPtr( rowIndex ) = j; 
        end
    end

    rowPtr(m+1) = length(column) + 1 ; 
    
    % check that there are not any empty rows
    for j = length(rowPtr):-1:1
        if rowPtr(j) == 0
            error('empty rows are not allowed for current format') ; 
        end
    end

end



function [row column valuesOrig] = sortTriples(row, column, valuesOrig, left, right)

% 	 Simple quick sort. 
% 	 Taken directly from "The C Programming Language", Kernighan and Ritchie.
% 	 Sort triples into increasing lexicographical order by (row, column, value). 
% 	 
% 	 Input:
% 		int row                 Row array. 
% 		int column              Column array. 
% 		double valuesOrig		Values array. 
% 		int left				First index of array to sort. 
% 		int right				Last index of array to sort. 
% 	 
% 	 Output:
% 		int row                 Row array, sorted in the relevant region. 
% 		int column          	Column array, sorted in the relevant region. 
% 		double valuesOrig		Values array, sorted in the relevant region. 
% 
%
%      Alex Kaiser, LBNL, 7/2010
%

    if (left >= right)
        return ; 
    end
    
    pivotIndex = floor( (left + right) / 2 ) ; 
    [row column valuesOrig] = swap(row, column, valuesOrig, left, pivotIndex) ; 
    last = left ; 
    
    for i = left+1 : right
        if compareTriple(row, column, valuesOrig, i, left) < 0 %row(i) < row(left)
            last = last + 1 ;
            [row column valuesOrig] = swap(row, column, valuesOrig, last, i) ;
        end
    end
    
    [row column valuesOrig] = swap(row, column, valuesOrig, left, last) ;

    [row column valuesOrig] = sortTriples(row, column, valuesOrig, left, last-1) ; 
    [row column valuesOrig] = sortTriples(row, column, valuesOrig, last + 1, right) ; 
    
end


function [row column valuesOrig] = swap(row, column, valuesOrig, i, j)
% 	 Swap entries in two positions of all three input arrays. 
% 	
% 	 Input:
% 		int row         		Row array. 
% 		int column          	Column array. 
% 		double valuesOrig		Values array. 
% 		int i					First index to swap.  
% 		int j					Second index to swap. 
%  
% 	 Output:
% 		int row    				Row array, swapped.  
% 		int column      		Column array, swapped.   
% 		double valuesOrig		Values array, swapped. 
% 
%
%      Alex Kaiser, LBNL, 7/2010
%

    temp = row(i) ; 
    row(i) = row(j) ; 
    row(j) = temp ; 

    temp = column(i) ; 
    column(i) = column(j) ; 
    column(j) = temp ; 
    
    temp = valuesOrig(i) ; 
    valuesOrig(i) = valuesOrig(j) ; 
    valuesOrig(j) = temp ; 
    
end


function res = compareTriple(row, column, valuesOrig, i, j)

%     	 Compares (row, col, val) triples in lexicographical order. 
% 	 
% 	 Input:
% 		int row[]				Row array. 
% 		int column[]			Column array. 
% 		double valuesOrig[]		Values array. 
% 		int i					First index to compare.  
% 		int j					Second index to compare.
% 
% 	 Output:
% 		int res (returned)		Value is 
% 									-1 if entry[i] < entry[j]
% 									 1 if entry[i] > entry[j]
% 									 0 if equal 
%
%         
%      Alex Kaiser, LBNL, 7/2010
%

    if row(i) < row(j)
        res = -1 ; 
    elseif row(i) > row(j) 
        res = 1 ; 
    else %if row(i) == row(j)
            if column(i) < column(j)
                res = -1 ; 
            elseif column(i) > column(j) 
                res = 1 ;
            else % if column(i) = column(j) 
                if valuesOrig(i) < valuesOrig(j)
                    res = -1; 
                elseif valuesOrig(i) < valuesOrig(j)
                    res = 1; 
                else %all values are equal
                    res = 0; 
                end
            end
    end
end


function sortedAndNoDuplicates = isSortedAndNoDuplicates(row, column, valuesOrig)

%   Checks whether a given array is sorted and duplicate free
%     
%     	 Input:
% 	 	int row[]				Row indices.
% 		int column[]			Column indices.
% 		double valuesOrig[]		Values of matrix entries.
% 		int origLength			Original length of array, which may contain duplicate entries.
% 
% 	Output:
% 	char (returned)  Whether array is sorted and duplicate free.
%         
%
%      Alex Kaiser, LBNL, 7/2010
%

    for j=1:length(row)-1
        
        % check for an out of order element
        if( compareTriple(row, column, valuesOrig, j+1, j) ~= 1 )
            sortedAndNoDuplicates = 0; 
            return; 
        else
            % check for a duplicate entry
            if( column(j+1) == column(j) && row(j+1) == row(j) ) 
                sortedAndNoDuplicates = 0; 
                return; 
            end
        end
        
    end
        
    sortedAndNoDuplicates = 1;
    return; 
        
end













    
    
    
    
    