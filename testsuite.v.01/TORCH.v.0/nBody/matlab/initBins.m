function [bins numBins] = initBins(size, cutoff)
% 	 Binning code for cutoff method.
% 
% 	 Input:
% 	 size               Boundary of region. 
%    cutoff             Force cuttoff. 
% 
% 	 Output:
% 	 bins               Array of bins for cutoff method.
% 	 int numBins         Length of the above array.
%     
%
%  Alex Kaiser, LBNL, 10/2010
%        
     
    numBins = ceil(size / cutoff) ; %check 
    
    bins(numBins).numParticles = 0;
    bins(numBins).parts = zeros(5,1); %list of particle indices
    
    
    %might need to fix sizing. 
    for i=1:numBins
        bins(i).numParticles = 0;
        bins(i).parts = zeros(5,1); %list of particle indices
    end
    
end