% Script to check 1D FFT. 
% 
% Randomly generates complex data, calculates fft and inverts fft. 
% Performs such checks three separate ways:
%     1. Cooley-Tukey FFT
%     2. Stockham FFT
%     3. Four Step FFT
% 
% Checks that RMS error is under specified tolerance for each case. 
% 
% Parameters:
%  	 int n				Length of input, must be power of two. 
%  	 int n1, n2			Integers such that n = n1 * n2. Each must be a power of two.
%    double tol         Tolerance for RMS error.       
% 
% Output:
%      Prints whether each RMS error is under tolerance. 
% 
% Alex Kaiser, LBNL, 9/2010
% 

disp('Check One dimensional FFT.'); 

if( ~(exist('spectralPass', 'var')))
    spectralPass = true ; 
end

n = 2^10 ; 

% Must set n1, n2 such that n = n2*n2. 
n1 = 2^5 ;  
n2 = 2^5 ; 

tol = 1e-10; 

a = rand(n, 1) + 1i * rand(n,1) ;

% compute Stockham FFT
disp('Check Stockham FFT:'); 
tic
res = stockhamFFT(a,n,-1) ;
toc

% invert and compare rms error
aInv = stockhamFFT(res, n, 1) *(1/n) ;

%calc rms error after inversion
rmsErr = rmsError(a, aInv); 
disp('rms error upon inversion = ');
disp( rmsErr ); 
if( abs(rmsErr) > tol )
    disp('RMS error too high. Stockham FFT test failed.'); 
    pass(1) = 0; 
else
    disp('Stockham FFT test passed.'); 
    disp(' '); 
    pass(1) = 1; 
end


disp('Check Cooley-Tukey FFT:'); 

% compute Cooley Tukey FFT
tic
res = cooleyTukeyFFT(a,n,-1) ;
toc

% invert 
aInv = cooleyTukeyFFT(res, n, 1) *(1/n) ;

%calc rms error after inversion
rmsErr = rmsError(a, aInv); 
disp('rms error upon inversion = ');
disp( rmsErr ); 
if( abs(rmsErr) > tol )
    disp('RMS error too high. Cooley Tukey FFT test failed.');
    pass(2) = 0;
else
    disp('Cooley Tukey FFT test passed.'); 
    disp(' '); 
    pass(2) = 1;
end


disp('Check Four-Step FFT:'); 

% compute Fourstep FFT
tic; 
res = fourStepFFT(a, n, n1, n2, -1) ;
toc; 


% invert and compare rms error
aInv = fourStepFFT(res, n, n1, n2, 1) *(1/n) ;

%calc rms error after inversion
rmsErr = rmsError(a, aInv); 
disp('rms error upon inversion = ');
disp( rmsErr ); 
if( abs(rmsErr) > tol )
    disp('RMS error too high. Four step FFT test failed.'); 
    pass(3) = 0;
else
    disp('Four step FFT test passed.'); 
    disp(' '); 
    pass(3) = 1;
end

%calculate max difference with built-in FFT function if desired
%maxBuiltInDiff = max(res - fft(a))

if( pass(1) && pass(2) && pass(3) )
    disp('1D FFT test passed.'); 
    spectralPass = spectralPass & true ;
else
    disp('1D FFT test failed.'); 
    spectralPass = false ;
end






