% Run all Matlab script. 
% Enters each directory and runs all check scripts. 
%
% Alex Kaiser, LBNL, 10/2010
%

mainDirectory = pwd; 
format long; 

tStart = tic ; 


disp('Dense Linear Algebra Tests.'); 
cd 'denseLinearAlgebra/matlab' ; 
checkLU ;
checkQR ;  
cd(mainDirectory) ; 
mainDirectory = pwd;


disp('Sparse Linear Algebra Tests.'); 
cd sparseLinearAlgebra/matlab ; 
checkSpmv ;
checkSpts ; 
checkCG ; 
checkGmres ; 
cd(mainDirectory) ;
mainDirectory = pwd;  


disp('Structured Grid  Tests.'); 
cd structured_grid/matlab ; 
check3DCentralDiffs; 
check3DCurl ; 
check3DDiv ; 
check3DGrad ; 
check3DLaplacian ; 
checkSpMVLaplacian ; 
check3DHeat ; 
check3DHeatVector ; 
check3DHeatImplicit ; 
check3DHeatImplicitVector ; 
check3DHeatSpectral ; 
checkMultiGrid ; 
cd(mainDirectory) ; 
mainDirectory = pwd;  


disp('Spectral Methods Tests.'); 
cd spectral_methods/matlab ; 
checkFFT ;
checkConvolution ; 
checkFFT3D ; 
cd(mainDirectory) ; 
mainDirectory = pwd;  


disp('N Body Tests.'); 
cd nBody/matlab ; 
checkAll ; 
cd(mainDirectory) ;  
mainDirectory = pwd;


disp('Monte Carlo Tests.'); 
cd monte_carlo/matlab ; 
checkQMCintegrate ; 
cd(mainDirectory) ;  
mainDirectory = pwd;


if denseLinearAlgebraPass 
    disp('Dense Linear Algebra tests passed.') ; 
else
    disp('Dense Linear Algebra tests failed.') ;
end


if sparseLinearAlgebraPass 
    disp('Sparse Linear Algebra tests passed.') ; 
else
    disp('Sparse Linear Algebra tests failed.') ;
end


if structuredGridPass 
    disp('Structured Grid tests passed.') ; 
else
    disp('Structured Grid tests failed.') ;
end


if spectralPass 
    disp('Spectral Methods tests passed.') ; 
else
    disp('Spectral Methods tests passed.') ;
end


if nBodyPass 
    disp('N Body tests passed.') ; 
else
    disp('N Body tests passed.') ;
end


if qMCintegratePass 
    disp('Quasi-Monte Carlo integrate tests passed.') ; 
else
    disp('Quasi-Monte Carlo integrate tests failed.') ;
end

fprintf(1,'\n') ; 
disp('Total time for all tests:'); 
toc(tStart); 
fprintf(1,'\n') ;


if ( denseLinearAlgebraPass && ...
     sparseLinearAlgebraPass && ...
     structuredGridPass && ...
     spectralPass && ...
     nBodyPass && ...
     qMCintegratePass  ) 
 
    fprintf(1, '\nAll tests passed.\n\n') ; 
else
    fprintf(1, '\nTests failed.\n\n') ;  
end

    

