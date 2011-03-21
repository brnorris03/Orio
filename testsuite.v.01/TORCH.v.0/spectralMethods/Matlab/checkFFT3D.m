% Script to check 3D FFT. 
% 
% Randomly generates complex data, calculates fft and inverts 3D fft. 
% Checks that RMS error is under specified tolerance after inversion. 
% 
% Parameters:
% 	 int m							Length of input in first dimension, must be power of two. 
% 	 int m1, m2						Integers such that m = m1 * m2. Each must be a power of two.
% 	 int n							Length of input in second dimension, must be power of two. 
% 	 int n1, n2						Integers such that n = n1 * n2. Each must be a power of two. 
% 	 int p							Length of input in third dimension, must be power of two. 
% 	 int p1, p2						Integers such that p = p1 * p2. Each must be a power of two.
%    double tol                     Tolerance for RMS error.       
% 
% Output:
%      Prints whether RMS error is under tolerance. 
% 
% Alex Kaiser, LBNL, 9/2010
% 


disp('Check 3D FFT.') ;

if( ~(exist('spectralPass', 'var')))
    spectralPass = true ; 
end

m = 2^5; 
m1 = 2^3; 
m2 = 2^2; 

n = 2^6 ; 
n1 = 2^3 ;  
n2 = 2^3 ; 

p = 2^4; 
p1 = 2^2; 
p2 = 2^2; 

tol = 1e-10; 

a = rand(m,n,p) + 1i * rand(m,n,p) ;


disp('3D FFT time:'); 

% calculate FFT
tic; 
res =  FFT3D(a,  m, m1, m2, n, n1, n2, p, p1, p2, -1) ;
toc; 

% invert
aInv = FFT3D(res,  m, m1, m2, n, n1, n2, p, p1, p2, 1) * (1/(n*m*p)) ;
rmsErr = rmsError3(a, aInv) ; 

disp('rms error upon inversion = ');
disp( rmsErr ); 
if( abs(rmsErr) > tol )
    disp('RMS error too high. 3D FFT test failed.');  
    specralPass = spectralPass & true ; 
else
    disp('3D FFT test passed.'); 
    disp(' '); 
    specralPass = false ; 
end


%compare with built-in FFT
%builtIn = fftn(a) ; 
%maxDiff = max( res(:) - builtIn(:) ) 