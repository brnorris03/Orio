function bins = assignBins(particles, bins, n, cutoff)
% 	 Assigns particles to bins for cutoff method.
% 
% 	 Input:
% 	 particles          Particles to assign.
% 	 bins               Bins in which to place particles.
% 	                         Must be pre-initialized by calling initBins.
% 	 n                  Number of particles.
%      cutoff             Distance for force cutofff. 
% 
% 	 Output:
% 	 bins               Bins with particles set.
%      
%  Alex Kaiser, LBNL, 10/2010
%  
%      
     
    for i = 1:n
        binNum = (ceil(particles(i).x / cutoff)) ;
        
        % trap for exactly zero case
        if(binNum == 0)
            binNum = 1; 
        end
        
        bins(binNum).numParticles = bins(binNum).numParticles + 1;
        k = bins(binNum).numParticles; 
        bins(binNum).parts(k) = i ; %adds i to the particle list  
    end

end