function W = c_to_rFFT( x, n, n1, n2, sign)
% 
% 	 Performs a complex to real FFT. 
% 	 Data must be correctly packed for transformation to invert properly.  
% 	 	 
% 	 Input:
% 	 complex vector x				Input array of length n/2 +  1, overwritten in function. 
% 										First and last elements must be real for input data to reperesent output of R->C FFT. 
% 	 int n							Length of original real data, must be power of two. 
% 	 int n1, n2						Integers such that n = n1 * n2. Each must be a power of two. 
% 	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
% 	 
% 	 Output:
% 	 struct complex * (returned)	FFT of x, interleaved as length n/2 complex.
%
%
% Alex Kaiser, LBNL, 9/2010. 
%


%algorithm statement uses zero index, add 1 to all indices
m = n/2; 
V = zeros(m,1); 

V(1) = (x(1) + x(m+1)) + 1i*(x(1) - x(m+1)) ;
V(n/4+1) = 2*(real(x(n/4 + 1)) - sign * 1i * imag(x(n/4 + 1))  ) ; 
                       

for k = 1:n/4-1
   dZero = x(k+1) + conj(x(m-k+1)) ;
   rootOfUnity = exp(sign * 2 * pi * 1i * k / n) ;
   dOne = 1i * rootOfUnity * (x(k+1) - conj( x(m-k+1) )) ;
   V(k+1) = (dZero + dOne); 
   V(m-k+1) = (conj(dZero) - conj(dOne)) ; 
end

W = fourStepFFT( V, m, (n1)/2, n2, sign); 

