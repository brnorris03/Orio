% 
%    Check symmetric QR factorization for computation of eigenvalues. 
%    
%      
% 	 Parameters:
% 	 double n               Dimension of matrix. 
% 	 double tolMax          Tolerance for max difference.
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

maxDiff = -1 ; 

% Check randomized matrices with known eigenvalue distributions. 
for t = 1:9
    
    fprintf(1, '\nTesting matirx of type %d.\n', t); 
    
    [a lambdas] = testRandomMatrix(n, t); 
    
    tic; 
    d = symmetricQR(a); 
    toc; 
    
    evals = sort(diag(d)); 
    generatedEvals = sort(lambdas) ; 
    differences = evals - generatedEvals; 

    % Output eigenvalues if desired.  
    %{
    disp('Implicit QR computed eigenvalues:'); 
    evals
    disp('Original generated eigenvalues:'); 
    generatedEvals 
    differences
    %}

    if max(abs(differences)) > tolMax
        fprintf(1,'High numerical difference found on matrix of type %d',t); 
    end
    
    if max(abs(differences)) > maxDiff
        maxDiff = max(abs(differences)); 
    end        
        
end

disp('Maximum difference between original and computed eigenvalues:');
maxDiff

if( maxDiff < tolMax)
    disp('QR factorization of randomized matrix with known eigenvalues max difference test passed.');
    denseLinearAlgebraPass = denseLinearAlgebraPass & true ;
else
    disp('QR factorization of randomized matrix with known eigenvalues max difference test failed.');
    denseLinearAlgebraPass = false ; 
end




% If desired, also test classical test matrices. 
% Verification here is compartison with built in Matlab function and 
% thus not preferred. 
% Default: Off. 

testClassicalMatrices = 0; 

if testClassicalMatrices

    
    maxDiff = -1 ; 

    % Check classical test matrices. 
    for t = 1:9

        [a lambdas] = testRandomMatrix(n, t); 

        d = symmetricQR(a); 
        evals = sort(diag(d)); 
        builtInEvals = sort(eig(a)); 
        differences = evals - builtInEvals;  

        % Output eigenvalues if desired.  
        %{
        disp('my implicit QR computed eigenvalues:'); 
        evals
        disp('matlab native eig output:'); 
        builtInEvals 
        differences
        %}

        if max(abs(differences)) > tolMax
            fprintf(1,'High numerical difference found on classical matrix of type %d\n',t); 
        end

        if max(abs(differences)) > maxDiff
            maxDiff = max(abs(differences)); 
        end        

    end

    disp('Maximum difference between computed and built in eigenvalues:');
    maxDiff

    if( maxDiff < tolMax)
        disp('QR factorization of classical matrix max difference test passed.'); 
    else
        disp('QR factorization of classical matrix max difference test failed.'); 
    end
    
end





