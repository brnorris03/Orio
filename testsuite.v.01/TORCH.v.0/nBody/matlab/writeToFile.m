function [] = writeToFile( file, n, particles)
% 
% Outputs positions of all particles to a file. 
% 
% Input:
%     file        File to print. 
%     n           Number of particles to print. 
%     particles   Array of particles. 
%     
% Output:
%     Position of particles written to file. 
%     
%     
% Alex Kaiser, LBNL, 10/2010
% 


    for i = 1:n
        fprintf(file, '%g %g\n', particles(i).x, particles(i).y ); 
    end
end