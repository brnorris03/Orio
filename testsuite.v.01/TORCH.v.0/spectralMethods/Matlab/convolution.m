function F = convolution(a, b, n, n1, n2)
% 
% 	 Computes a portion of the convolution of real vectors a and b naively. 
% 	 Provided only for verification.
% 	 
% 	 Input:
% 	 vector a				First vector for convolution. 
% 	 vector b				Second vector for convolution. 
% 	 int n					Total length of both vectors. 
%	 int n1, n2				Integers such that n = n1 * n2. Each must be a power of two. 
% 	 
% 	 Output:
% 	 vector F (returned)	The first numToCompute elements of the convolution. 
% 
% 
% Alex Kaiser, LBNL, 9/2010
%



twoN = 2*n; 
twoN1 = 2*n1; 
% vectors to be zero padded for calc
% double n1 to maintain condition that n = n1*n2

a = [ a; zeros(n,1)]; 
b = [ b; zeros(n,1)]; 

A = r_to_cFFT(a, twoN, twoN1, n2, -1);
B = r_to_cFFT(b, twoN, twoN1, n2, -1);

c = A .* B ;

c_to_rFFT( c, twoN, twoN1, n2, 1) ;

result = (1/twoN) * c_to_rFFT( c, twoN, twoN1, n2, 1) ;

F = zeros(n, 1); 
for k = 0:n-1
    F(2*k+1) = real( result(k+1) ) ; 
    F(2*k+2) = imag( result(k+1) ) ; 
end
