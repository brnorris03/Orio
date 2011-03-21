function res = boxInt(n, s)
% 
% 	 Crandall's Box Sum scheme. 
% 	 Computes B_n(s) via evaluation of an infinite series. 
% 	
% 	 See Richard Crandall's "Theory of Box Series"
% 	 
% 	 Input:
% 			int n					Dimension to evaluate
% 			double s				Parameter for function
% 	 
% 	 Output:
% 			double (returned)		Value of the function
%
%   
%   Alex Kaiser, LBNL, 9/2010. 
%

m = n-1; 
gamma = ones(m,1); 
A = 1; 
p = 1; 

t = 2/n; 

k = 1 ; 

while true
    
    sigma = k-1-s/2; 
    gamma(1) = sigma/(1 + 2*k) * gamma(1); 
    
    for mu=2:m
       gamma(mu) = (sigma * gamma(mu) + gamma(mu-1) ) / (1 + 2*k/mu) ; 
    end
    
    p = p*t; 
    A = A + gamma(m)*p ; 
    
    if (abs(gamma(m)*p) < 1e-14)
        break
    end
    
    k = k+1; 
end


res = A * n^(1 + s/2) / (s + n) ; 

