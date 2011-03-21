function pass = verifyNBody3d(guessLog, trueLog, tol)
% 
% Computes the RMS error between each component of two force logs. 
% Ensures that each is below the given tolerance. 
% 
% Input:
% guessLog         Force log for force to verify. 
% trueLog          Force log for true force. 
% tol              Maximum difference between two forces. 
% 
% Output:
% returns pass     True if each component is under tolerance. 
% 
% 
% Alex Kaiser, LBNL, 10/2010
% 


maxErrX = 0; 
maxErrY = 0;
maxErrZ = 0;

for i = 1:guessLog.steps
    currentErrX = rmsError( guessLog.x(:,i), trueLog.x(:,i) ) ; 
    if currentErrX > maxErrX
        maxErrX = currentErrX ;
    end
    
    currentErrY = rmsError( guessLog.y(:,i), trueLog.y(:,i) ) ; 
    if currentErrY > maxErrY
        maxErrY = currentErrY ;
    end
    
    currentErrZ = rmsError( guessLog.z(:,i), trueLog.z(:,i) ) ; 
    if currentErrZ > maxErrZ
        maxErrZ = currentErrZ ;
    end

end

maxErrX
maxErrY
maxErrZ

if (maxErrX < tol) && (maxErrY < tol) && (maxErrZ < tol)
    disp('RMS error below tolerance.'); 
    disp('Test passed.'); 
    pass = true; 
else
    disp('RMS error above tolerance.'); 
    disp('Test failed.'); 
    pass = false;
end


