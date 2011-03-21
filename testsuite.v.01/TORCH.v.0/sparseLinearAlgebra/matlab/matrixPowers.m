function B = matrixPowers(x, n, rowPtr, columnIndices, values, k)

% naive computation of matrix powers

B = zeros(n,k); 

B(:,1) = x; 

for i=2:k
    B(:,i) = spmv( B(:,i-1), n, n, rowPtr, columnIndices, values); 
end






