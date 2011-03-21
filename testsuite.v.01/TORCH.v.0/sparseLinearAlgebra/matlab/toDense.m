function a = toDense(m, n, rowPtr, columnIndices, values) 

% Converts a CSR format matrix to standard dense matrix
% For debugging and checking use.
% 
% Input:
%      int m                   Matrix dimension, number of rows
%      int n                   Matrix dimension, number of columns
%      int rowPtr              Row pointer vector
%      int columnIndices       Column indices vector
%      double values           Values of matrix entries
% 
% Output:
%      a                       Dense matrix with same values as input matrix
%     
%     Alex Kaiser, LBNL, 7/2010 


a = zeros(m,n); 

for i = 1:length(rowPtr)-1
    
    for jj = rowPtr(i):rowPtr(i+1)-1
        a(i, columnIndices(jj)) = values(jj) ;
    end
    
end



