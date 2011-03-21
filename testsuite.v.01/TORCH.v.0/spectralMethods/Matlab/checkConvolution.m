% 
% 	 Checks convolution of real, integer data. 
% 	 Input is randomly generated in the range [0, 2^b - 1]. 
% 	 A bound for the validity of the verification scheme is checked, namely that:
% 		2*b + log2(n) <= 53. 
% 	 All output data must be within the specified tolerance of an integer for test to pass. 
% 	 The first 'numToCompute' elements of the convolution will be also computed with a naive convolution algorithm. 
% 	 Output must match these numbers also within the specified tolerance. 
% 	 
% 	 Parameters:
% 	 int n						Length of the convolution to perform.
% 	 int n1, n2					Integers such that n = n1 * n2. 
% 	 int b						Bound for random number generator. 
% 	 
% 	 Output:
% 	 Prints whether test passed and information about numerical differences. 
%      
     

disp('Checking convolution.'); 

if( ~(exist('spectralPass', 'var')))
    spectralPass = true ; 
end


tol = 10e-5; 

n = 2^13 ; 
n1 = 2^7 ;  
n2 = 2^6 ; 

b = 10; 

a = getRandomInt(n,1,b); 
b = getRandomInt(n,1,b); 

disp('Time for convolution with fft:'); 
tic
F = convolution(a, b, n, n1, n2); 
toc
disp(' '); 

if(2*b + log2(n) > 53)
    disp('2*b + log2(n) must be less than or equal to 53 for verification scheme to be valid.') ;
    disp('Test failed'); 
    return ; 
end

% check that all elements of result are integers within tolerance
passed = 1 ; 
maxDiff = 0; 
for j=1:n
    
    if( abs(round(F(j)) - F(j)) > tol )
        fprintf(1,'F(%d) is more than the tolerance of %e from the nearest integer.\n', j, tol) ;  
        passed = 0; 
    end
    
    if( abs(round(F(j)) - F(j)) > maxDiff )
        maxDiff = abs(round(F(j)) - F(j)) ; 
    end
    
end

disp('Max difference with integer data:'); 
disp(maxDiff); 

if passed
    disp('Convolution test of comparison with integers on convolution passed') ; 
else
    disp('Convolution test failed on integer comparisons.'); 
end



numToCompute = min(50, n); 
naiveConv = naiveConvolution(a, b, n, numToCompute); 

maxDiff = 0; 

for j=1:numToCompute
    
    if (abs(naiveConv(j) - F(j)) > tol)
        fprintf(1,'element %d of naive convolution does not match convolution output.\n', j); 
        passed = 0; 
    end
    
    if (abs(naiveConv(j) - F(j)) > maxDiff)
        maxDiff = abs(naiveConv(j) - F(j)) > tol ; 
    end
    
end

disp('Max difference with naive convolution data:'); 
disp(maxDiff); 

if passed 
    disp('Convolution test passed.'); 
else
    error('Convolution test failed.'); 
end

specralPass = spectralPass & passed ; 


