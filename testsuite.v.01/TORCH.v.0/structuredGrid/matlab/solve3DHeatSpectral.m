function soln = solve3DHeatSpectral(dt, tSteps, l, h)

%   3D heat equation solve for fixed initial conditions. 
%   Spectral method. 
% 
% 
%   input:
% 
%      double dt          Time step for solve 
%      int tSteps         total number of timesteps to perform
%      double l           width of cube, same on all dimensions  
%      double h           grid height
% 
%   output:
% 
%     soln    4D real     solution to heat equation. indexed (x,y,z,t)
% 
%   
%   Alex Kaiser, LBNL, 7/2010


%use fixed initial conditions for now
f = @(x,y,z) sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z);

%lambda = dt / (h*h) ; 

N = l/h - 1;  %don't include ghost zones in this number
M = tSteps ;  
soln = zeros(N,N,N,M); % don't include the ghost zones here


%set initial conditions
%ghost zones added automatically in init function 
initTemp = initGrid3D(f, h, l, l, l); 

%remove ghost zones, as they are zero and not needed
initTemp = initTemp(2:N+1, 2:N+1, 2:N+1); 
soln(:,:,:,1) = initTemp(:,:,:); 


% gather parameter values for FFT routines
log2Length = floor(log2(N)); 

if mod(log2Length,2) == 1
    exp1 = floor(log2Length / 2 ) + 1; 
    exp2 = floor(log2Length / 2 );
else
    exp1 = log2Length / 2 ; 
    exp2 = log2Length / 2 ;
end

n1 = 2^exp1 ; 
n2 = 2^exp2 ; 

if (n1 * n2 ~= N)
    error('Problems in factorization size. Size must be power of two. Exiting.'); 
end

% use hand coded routine
initFFT = FFT3D(soln(:,:,:,1), N, n1, n2, N, n1, n2, N, n1, n2, -1) ; 

% or built in matlab
%initFFT = fftn(soln(:,:,:,1)); 


    %compute the exponential lookup table for advancing data 
    twiddles = zeros(N,N,N); 

    %use zero indexed loops
    %add one to index computations. 
    for k = 0:N-1
        for j = 0:N-1
            for i = 0:N-1

                if i < N/2
                    iBar = i; 
                else
                    iBar = i-N; 
                end

                if j < N/2
                    jBar = j; 
                else
                    jBar = j-N; 
                end

                if k < N/2
                    kBar = k; 
                else
                    kBar = k-N; 
                end                

                twiddles(i+1, j+1, k+1) = exp(-4*pi^2 * (iBar^2 + jBar^2 + kBar^2));  
            end
        end
    end


t = 0; 
for j = 2:tSteps    
   t = t + dt; 
   soln(:,:,:,j) = (twiddles .^ t) .* initFFT ; 
   
   % use hand coded routine
   soln(:,:,:,j) = (1.0/(N*N*N)) * FFT3D(soln(:,:,:,j), N, n1, n2, N, n1, n2, N, n1, n2, 1) ;
   
   % or built in matlab
   %soln(:,:,:,j) = ifftn(soln(:,:,:,j)) ; 
   j
end
 

