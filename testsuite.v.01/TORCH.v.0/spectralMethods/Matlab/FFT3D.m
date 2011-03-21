function res = FFT3D( x, m, m1, m2, n, n1, n2, p, p1, p2, sign)
% 
% 	 Three dimensional FFT. 
% 	 To calculate the inverse FFT, use sign = 1 and devide output by m*n*p after function call.
% 	 
% 	 Input:
% 	 complex 3D array x     		Input array, overwritten in function. 
% 	 int m							Length of input in first dimension, must be power of two. 
% 	 int m1, m2						Integers such that m = m1 * m2. Each must be a power of two.
% 	 int n							Length of input in second dimension, must be power of two. 
% 	 int n1, n2						Integers such that n = n1 * n2. Each must be a power of two. 
% 	 int p							Length of input in third dimension, must be power of two. 
% 	 int p1, p2						Integers such that p = p1 * p2. Each must be a power of two.
% 	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
% 	 
% 	 Output:
% 	 complex 3D array (returned)	FFT of x.
%      
%      
%  Alex Kaiser, LBNL, 9/2010
% 



% take the transformations down each column
% no need for a temporary vector here, because columns are contiguous in
% memory
for j = 1:n
    for k = 1:p 
        x(:,j,k) = fourStepFFT( x(:,j,k), m, m1, m2, sign); 
    end
end

%take transformations across each row
for j = 1:m
    for k = 1:p 
        temp = x(j,:,k) ; 
        x(j,:,k) = fourStepFFT(temp, n, n1, n2, sign); 
    end
end

%take transformations down each depth
for j = 1:m
    for k = 1:n 
        temp = x(j,k,:) ; 
        x(j,k,:) = fourStepFFT(temp , p, p1, p2, sign); 
    end
end

res = x;
