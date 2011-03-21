function [rowPtr columnIndices values] = get3DLaplacianMatrix(n, height)

%   Returns an n x n x n heat equation matrix for a 3D homogeneous Laplacian. 
%   Assumes problem is homogeneous and ignores boundaries accordingly. 
% 
%   Input:
%         int n           Dimension of original heat equation grid. 
%                            Matrix will be of dimension n^3 by n^3
%         double height   Constant for heat equation solves. lambda = dt / (h*h)
%     
%   Output:
%         int rowPtr              Row pointer vector
%         int columnIndices       Column indices vector
%         double values           Values of matrix entries
%         
%
%      Alex Kaiser, LBNL, 7/2010
%


nnzSoFar = 0; 

nCubed = n * n * n ; 
row = zeros(7*nCubed, 1); 
column = zeros(7*nCubed, 1); 
valuesOrig = zeros(7*nCubed, 1); 

lambda = 1 / (height * height) ; 


% this would traverse an n x n x n 3D grid array...
for i = 1:n
    for j = 1:n
        for k = 1:n
            
            diagIndex = k + n*(j-1) + n*n*(i-1) ; 
            
            if i ~= 1
                nnzSoFar = nnzSoFar + 1; 
                row(nnzSoFar) = diagIndex ; 
                column(nnzSoFar) = diagIndex - n*n; 
                valuesOrig(nnzSoFar) = lambda ; 
            end
            
            if j ~= 1
                nnzSoFar = nnzSoFar + 1; 
                row(nnzSoFar) = diagIndex ; 
                column(nnzSoFar) = diagIndex - n; 
                valuesOrig(nnzSoFar) = lambda ; 
            end
            
            if k ~= 1
                nnzSoFar = nnzSoFar + 1; 
                row(nnzSoFar) = diagIndex ; 
                column(nnzSoFar) = diagIndex - 1; 
                valuesOrig(nnzSoFar) = lambda ; 
            end
            
            
            %always place the diagonal element
            nnzSoFar = nnzSoFar + 1; 
            row(nnzSoFar) = diagIndex ; 
            column(nnzSoFar) = diagIndex ; 
            valuesOrig(nnzSoFar) = - 6 * lambda ; 
            
            if k ~= n
                nnzSoFar = nnzSoFar + 1; 
                row(nnzSoFar) = diagIndex ; 
                column(nnzSoFar) = diagIndex + 1; 
                valuesOrig(nnzSoFar) = lambda ; 
            end
            
            if j ~= n
                nnzSoFar = nnzSoFar + 1; 
                row(nnzSoFar) = diagIndex ; 
                column(nnzSoFar) = diagIndex + n; 
                valuesOrig(nnzSoFar) = lambda ; 
            end
            
            if i ~= n
                nnzSoFar = nnzSoFar + 1; 
                row(nnzSoFar) = diagIndex ; 
                column(nnzSoFar) = diagIndex + n*n; 
                valuesOrig(nnzSoFar) = lambda ; 
            end
            
        end
    end
end


row = row(1:nnzSoFar) ;
column = column(1:nnzSoFar) ;
valuesOrig = valuesOrig(1:nnzSoFar) ;   


[rowPtr columnIndices values] = getCSRfromRowColumn(nCubed, nCubed, row, column, valuesOrig); 



