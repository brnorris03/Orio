function res = boxIntegrand( x, n, s )
% 
% 	 Computes the value of the integrand of B_n(s).
% 	 See sources for additional information on this family of functions. 
% 	
% 	 Input:
% 			vector x				Point at which to evaluate function. 
% 									Length of vector = Dimension. 
%           int n                   Dimension. 
% 			double s				Parameter for function
% 	 
% 	 Output:
% 			double (returned)		Value of the function
% 
%
%   Alex Kaiser, LBNL, 9/2010. 
%


res = 0.0; 

for i = 1:n
    res = res + x(i)*x(i) ; 
end

res = res ^ (s/2); 


