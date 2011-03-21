function bins = cleanBins(bins, numBins)
% 	 Removes all particles from bins.
% 
% 	 Input:
% 	 bins                        Bins in which to place particles.
% 	                                 Must be pre-initialized by calling initBins.
% 	                                 Does not modify particle pointers, only numParticles.
% 	                                 Use care that this parameter is correctly modified.
% 	 int numBins                 Number of bins.
% 
% 	 Output:
% 	 bins                        Bins with particles removed.
%      
%  Alex Kasier, LBNL, 10/2010
%  
    
    for i=1:numBins
        bins(i).numParticles = 0;
        bins(i).parts = zeros(5,1); %list of particle indices
    end
    
end