function y = spmv(x, m, n, rowPtr, columnIndices, values) 
% 
% 	 Sparse matrix vector multiply. 
%    Matrix in CSR format
% 	 Computes y = A * x. 
% 	 
% 	 
% 	 Input:
%         double x                Vector to multiply by
%         int m                   Matrix dimension, number of rows
%         int n                   Matrix dimension, number of columns
%         int rowPtr              Row pointer vector
%         int columnIndices       Column indices vector
%         double values           Values of matrix entries
% 		
% 	 
% 	 Output:
%         double y                The matrix-vector product
%         
%         
%      Alex Kaiser, LBNL, 7/2010
%
        

y = zeros(m,1); 

for rowNum = 1:m
   
   yTemp = 0.0; 
   
   for j = rowPtr(rowNum) : rowPtr(rowNum+1)-1
       
       %allows multiplication by a subarray without cropping
       if columnIndices(j) > n
           break
       end
       
       yTemp = yTemp + values(j) * x(columnIndices(j)) ; 
   end
    
   y(rowNum) = yTemp; 
end




