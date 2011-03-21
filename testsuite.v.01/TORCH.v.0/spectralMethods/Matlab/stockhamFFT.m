function y = stockhamFFT( x , n, sign) 
% 
% 	 Computes FFT of x using the Stockham FFT algorithm. 
% 	 Secondary one-dimensional FFT called by fourStep fft.
% 	 
% 	 Input:
% 	 complex vector *x				Input array, overwritten in function. 
% 	 int n							Length of input, must be power of two. 
% 	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
% 	 
% 	 Output:
% 	 complex vector (returned)      FFT of x. 
%
%
% Alex Kaiser, LBNL, 9/2010
%



% Since algorithm statement uses zero indexing, 
% loops follow this convention. Add one to every index computation

% this uses the 'second variant' of the loop organization

l = n/2; 
m = 1; 
y = zeros( size(x)); 

for iteration = 1:log2(n)
    
    for j = 0:l-1
        % compute w on the fly
        w = exp( j * sign * 2 * pi * 1i / (2*l) )  ; 
        for k = 0:m-1
            c0 = x(k + j*m + 1); 
            c1 = x(k + j*m + l*m + 1);
            y(k + 2*j*m + 1) = c0 + c1; 
            y(k + 2*j*m + m + 1) = w * (c0 - c1); 
            
        end
    end
    
    x = y; %for efficiency, should swap x and y
    
    %update l and m for next iteration
    l = l/2;   
    m = m*2; 
end


