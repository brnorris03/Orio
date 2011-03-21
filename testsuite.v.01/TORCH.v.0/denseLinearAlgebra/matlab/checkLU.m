% 
%    Check LU factorization. 
%    Max difference of the product of the factors and the original matrix 
%       must be under specified tolerance to pass. 
%      
% 	 Parameters:
% 	 double n               Dimension of matrix. 
% 	 double tolMax          Tolerance for max difference.
%    vector condArray       Array of parameters for influencing condition number of matrix.
%                           Checks cond = 1 and cond = 1 / sqrt(eps) by default. 
%                           cond = 1 / eps is optional and by default off, 
%                               as test normally fails for the large condition numbers that result. 
%                           Change boundaries on first loop to turn on. 
%
%                           Initial array:
%                               condArray = [10.0, 1/sqrt(eps), eps ] ;
%
%
% 	 Output:
% 	 Prints whether all norms are under supplied tolerances.
%
%    Alex Kaiser, LBNL, 7/2010
%


if( ~(exist('denseLinearAlgebraPass', 'var')))
    denseLinearAlgebraPass = true ; 
end

n = 200; 
tolMax = 1e-10; 
condArray = [10.0, 1/sqrt(eps), eps ] ; 

% generator severely needs updating 
% a = rand(n) ; 

pass = true ; 


for j = 1:length(condArray) - 1 %remove this -1 to include difficult condition numbers. 
    
    cond = condArray(j) ; 
    
    for type = 1:5

        a = getCondNumberMatrix(n, cond, type) ; 

        tic; 
        [ll uu] = LUFactorize(a) ;
        toc; 

        disp('Maximum difference for current factorization:'); 
        maxDiff = max(max(abs(ll*uu - a)))

        if( maxDiff < tolMax)
            fprintf(1, 'LU factorization max difference test passed on matrix of type %d.\n', type); 
        else
            fprintf(1, 'LU factorization max difference test failed on matrix of type %d.\n', type); 
            pass = false ; 
        end

    end

end

disp(' '); 

if pass
    disp('LU factorization max difference tests passed.'); 
    denseLinearAlgebraPass = denseLinearAlgebraPass & true ;
else
    disp('LU factorization max difference tests failed.');
    denseLinearAlgebraPass = false ;
end