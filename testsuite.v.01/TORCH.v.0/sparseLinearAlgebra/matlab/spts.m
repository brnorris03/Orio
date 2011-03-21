function x = spts(b, n, rowPtr, columnIndices, values) 

% 
% 	 Sparse lower triangular solve. 
%    Matrix in CSR format.
% 	 Solves A * x = b. 
%    Assumes that matrix is non-singular. 
% 	 
% 	 
% 	 Input:
%         double b                Right hand side of system of equations. 
%         int n                   Matrix dimension
%         int rowPtr              Row pointer vector
%         int columnIndices       Column indices vector
%         double values           Values of matrix entries
% 		
% 	 
% 	 Output:
%         double y                The matrix-vector product
%         
%         
%
%    Source - http://netlib.org/linalg/html_templates/node102.html
%
%      Alex Kaiser, LBNL, 7/2010
%

x = zeros(n,1); 

x(1) = b(1) / values(1); 

for rowNum = 2:n
    sum = 0.0; 
    
    for j = rowPtr(rowNum) : rowPtr(rowNum+1) - 2
        sum = sum + values(j) * x(columnIndices(j)) ; 
    end
    
    x(rowNum) = (b(rowNum) - sum) / values(rowPtr(rowNum+1) - 1) ; 
end








