function y = r_to_cFFT( x, n, n1, n2, sign)
% 
% 	 Performs FFT of real data, interleaving data and using a Four Step FFT after doing so. 
% 	
% 	 Input:
% 	 vector x						Real input array, overwritten in function. 
% 	 int n							Length of input, must be power of two. 
% 	 int n1, n2						Integers such that n = n1 * n2. Each must be a power of two. 
% 	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
% 	 
% 	 Output:
% 	 complex vector (returned)  	FFT of x, interleaved as length n/2 complex. 
% 
%      
%  Alex Kaiser, LBNL, 9/2010
%  
 


%algorithm statement uses zero index, add 1 to all indices
m = n/2; 

v = zeros(m,1); 
for j = 0:m-1
    v(j+1) = x(2*j + 1) + 1i * x(2*j + 2); 
end

w = fourStepFFT(v, m, (n1)/2, n2, sign) ; 

y = zeros(m+1,1); 

y(1) =  ( real(w(1)) + imag(w(1)) ) ;
y(n/4 + 1) =  ( real(w(n/4 + 1)) + sign * 1i * imag(w(n/4 + 1)) ) ; 
y(n/2 + 1) =  ( real(w(1)) - imag(w(1)) );

for j = 1:n/4-1
    dZero = w(j+1) + conj(w(m-j+1)) ; 
    rootOfUnity = exp(sign * 2 * pi * 1i * j / n) ;
    dOne = -1 * 1i * rootOfUnity * ( w(j+1) - conj(w(m-j+1)) ) ;
    y(j+1) = .5 * (dZero + dOne); 
    y(m-j+1) = .5 * (conj(dZero) - conj(dOne)) ; 
end


