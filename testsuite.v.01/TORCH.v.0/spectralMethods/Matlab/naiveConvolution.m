function F = naiveConvolution(a, b, n, numToCompute)
% 
% 	 Computes a portion of the convolution of real vectors a and b naively. 
% 	 Provided only for verification.
% 	 
% 	 Input:
% 	 vector a				First vector for convolution. 
% 	 vector b				Second vector for convolution. 
% 	 int n					Total length of both vectors. 
% 	 int numToCompute		Number of elements to compute. Only computes the first numToCompute 
% 								elements of the convolution for efficiency. 
% 	 
% 	 Output:
% 	 vector F (returned)	The first numToCompute elements of the convolution. 
% 
% 
% Alex Kaiser, LBNL, 9/2010
%



if 2*n < numToCompute
    disp('Can only compute 2*n convolution elements. Dafaulting to 2*n.'); 
    numToCompute = 2*n; 
end


F = zeros(numToCompute,1) ; 


% vectors to be zero padded for calc
% vector n1 to maintain condition that n = n1*n2

a = [ a; zeros(n,1)]; 
b = [ b; zeros(n,1)]; 

%for k = 0:2*n - 1
for k = 0:numToCompute - 1
    for j = 0:2*n - 1
        if (k - j) < 0
            subscript = k - j + 2*n ; 
        else
            subscript = k-j; 
        end
        F(k+1) = F(k+1) + a(j+1) * b(subscript+1) ;  
    end
end




