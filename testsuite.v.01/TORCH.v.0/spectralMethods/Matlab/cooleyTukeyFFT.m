function x = cooleyTukeyFFT(x, n, sign)
% 
% 	 Cooley-Tukey FFT
% 	 Algorithm 1.6.1 in Van Loan's 'Computational Frameworks for the Fast Fourier Transform'
% 	 Secondary one-dimensional FFT called by fourStep fft.
% 	 
% 	 Input:
% 	 complex vector x				Input array, overwritten in function. 
% 	 int n							Length of input, must be power of two. 
% 	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
% 	 
% 	 Output:
% 	 complex vector (returned)      FFT of x. 
%
%
% Alex Kaiser, LBNL, 9/2010
%

    t = log2(n); 
    x = permuteByBitReversal(x, n, t) ; 
    
    for q = 1:t
        
        L = 2^q ; 
        r = n/L ; 
        lStar = L/2 ; 
        
        for j = 0:lStar-1
           w = cos(2 * pi * j / L) + sign * 1i * sin(2 * pi * j / L); 
           
           for k = 0:r-1
              tau = w * x(k*L + (j+1) + lStar); 
              x(k*L + (j+1) + lStar) = x(k*L + (j+1)) - tau; 
              x(k*L + (j+1)) = x(k*L + (j+1)) + tau; 
           end
        end 
    end
end


function x = permuteByBitReversal(x, n, t)
    % Permute data by bit reversal. 
    % Algorithm 1.5.2 in Van Loan's 'Computational Frameworks for the 
    % Fast Fourier Transform'
    %
    % Input:
    % complex vector x      Vector to permute
    % int n                 Length
    % int t                 Constant to generating permutation index. 
    %
    % Output:
    % complex vector x      The permuted vector.        

    for k=0:n-1
        j = bitReverseIndex(k, t); 
        if(j > k)
            temp = x(j+1); 
            x(j+1) = x(k+1); 
            x(k+1) = temp; 
        end
    end 
end




function j = bitReverseIndex(k, t)
    % Gets index for bit reversal. 
    % Algorithm 1.5.1 in Van Loan's 'Computational Frameworks for the 
    % Fast Fourier Transform'
    %
    % Input:
    % int k                 Constant to generating permutation index.
    % int t                 Constant to generating permutation index.
    %
    % Output:
    % int j                 Permutation index. 
    
    j=0; 
    m=k; 
    for q = 0:t-1
       s = floor(m/2); 
       j = 2*j + (m - 2*s); 
       m = s; 
    end
end



