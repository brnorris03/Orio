function res = fourStepFFT( x, n, n1, n2, sign)
%
% 	 Computes FFT(x) by the four step method.
% 	 Primary one-dimensional FFT algorithm. 
% 	 To calculate the inverse FFT, use sign = 1 and devide output by n after function call.
% 	 
% 	 Input:
% 	 complex vector x				Input array, overwritten in function. 
% 	 int n							Length of input, must be power of two. 
% 	 int n1, n2						Integers such that n = n1 * n2. Each must be a power of two. 
% 	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
% 	 
% 	 Output:
% 	 complex vector (returned)  	FFT of x. 
%
%
% Alex Kaiser, LBNL, 9/2010
%

res = reshape(x, n1, n2) ; 

for j = 1:n1
    %force to take a contiguous vector
    temp = res(j,1:n2); 
    res(j,1:n2) = stockhamFFT( temp, n2, sign); 
end

twiddles = zeros(n1, n2); 
for j = 0:n1-1
    for k = 0:n2-1
        twiddles(j+1, k+1) = exp( sign * 2 * pi * 1i * j * k / n); 
    end
end

res = res .* twiddles; 

% NOTE - built in matlab ' operator performs conjugate transpose.
% use .' for simple transpose.   
res = res.' ; 

for j = 1:n2
    temp = res(j,1:n1); 
    res(j,1:n1) = stockhamFFT( temp, n1, sign); 
end

res = reshape(res, n, 1) ; 



