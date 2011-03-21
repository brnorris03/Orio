function [ forceLog ] = runCutoff3d(n, numSteps, forceType, output, saveFreq, verify) 
%
%    Source - http://www.cs.berkeley.edu/~volkov/cs267.sp09/hw2/
% 	 Runs simulation of N-Body problem using naive algorithm.
% 
% 	 Input:
% 	     int n                  Number of particles to simulate.
% 	     int numSteps           Number of steps to simulate.
% 	     int forceType          Which force to use.
% 	     bool output            If true, this routine will output a text file of the positions of the particles.
% 	     int saveFreq           if (output) data is written every this many time steps.
% 	     cool verify            If true, this routine store forces for verification purposes
% 
% 	 Output:
% 	     if( output ), a text file "naive.txt" is written with the positions of the particles at each step.
% 	     if( verify ), the arrays in the structure forceLog will be filled with force data for comparisons
%
% Alex Kaiser, LBNL, 10/2010
%


    %constants from common.cpp file
    density = .0005; 
    mass = .2; 
    cutoff = .01; 
    min_r = cutoff / 100; 
    dt = .0005;
    

    size = setSize(n, density); 
    
    if verify
        forceLog.n = n ; 
        forceLog.steps = numSteps ; 
        forceLog.x = zeros(n, numSteps) ; 
        forceLog.y = zeros(n, numSteps) ;
        forceLog.z = zeros(n, numSteps) ;
    else
        forceLog = 0; 
    end
    
    if (output) 
        file = fopen('cutoff3d.txt', 'w'); 
        fprintf(file, '%d %g\n', n, size); 
    end
    
    particles = initParticles3d(n, size, mass); 
    [bins numBins] = initBins(size, cutoff); 
    
    
    for step = 1:numSteps
        
        bins = assignBins(particles, bins, n, cutoff) ; 
        
        for i=1:n    
           particles(i).ax = 0; 
           particles(i).ay = 0;
           particles(i).az = 0;
        end     
        
        
        %compute forces
        for k = 1:numBins    
            
            %comare to local particles
            
            for i = 1:bins(k).numParticles
                for j = 1:bins(k).numParticles
                    localIndex = bins(k).parts(i); 
                    neighborIndex = bins(k).parts(j); 
         
                    if localIndex ~= neighborIndex
                        particles(localIndex) = applyForce3d(particles(localIndex), particles(neighborIndex) , cutoff, min_r, forceType); 
                    end
                end
            end
            
            %compare bin on left
            
            if k > 1                  
                leftBin = bins(k-1); 
                for i = 1:bins(k).numParticles
                    for j = 1:leftBin.numParticles
                        localIndex = bins(k).parts(i); 
                        neighborIndex = leftBin.parts(j); 

                        if localIndex ~= neighborIndex
                            particles(localIndex) = applyForce3d(particles(localIndex), particles(neighborIndex) , cutoff, min_r, forceType); 
                        end
                    end
                end
            end
            
            %compare bin on right
            if k < numBins 
                rightBin = bins(k+1); 
                for i = 1:bins(k).numParticles
                    for j = 1:rightBin.numParticles
                        localIndex = bins(k).parts(i); 
                        neighborIndex = rightBin.parts(j); 

                        if localIndex ~= neighborIndex
                            particles(localIndex) = applyForce3d(particles(localIndex), particles(neighborIndex) , cutoff, min_r, forceType); 
                        end
                    end
                end
            end     
             
        end  % end of force loop
        
        
        if verify 
            for i = 1:n
                forceLog.x(i,step) = particles(i).mass * particles(i).ax ; 
                forceLog.y(i,step) = particles(i).mass * particles(i).ay ;
                forceLog.z(i,step) = particles(i).mass * particles(i).az ;
            end
        end
        
        %move particles
        for i = 1:n
            particles(i) = move3d(particles(i), size, dt, forceType); 
        end
        
        if (output) && (mod(step, saveFreq) == 0)
            writeToFile3d(file, n, particles) ;
        end
        
        %empty bins
        bins = cleanBins(bins, numBins); 
    end
    
    
    
    if(output)
        fclose(file); 
    end
    

end