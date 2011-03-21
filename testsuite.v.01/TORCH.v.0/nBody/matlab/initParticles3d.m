function particles = initParticles3d(n, size, mass)
% 	 Initialize the particle positions and velocities.
% 	 Current implementation resets random number generator bcnrand for reproducible results.
% 
% 	 Input:
% 	 int n              Number of particles.
% 	 particle_t *p      Array of particles, must be preallocated.
% 
% 	 Parameters:
% 	 double velocityCoefficient    Velocities are set uniformly in the interval:
% 	                                   ( -velocityCoefficient, velocityCoefficient )
% 
% 	 Output:
% 	 Particles are set to randomized initial positions and velocities.
%      
%  Alex Kaiser, LBNL, 10/2010
%  
%  

    % allocate by initializing element with max index
    particles(n).x = 0;
    particles(n).y = 0;
    particles(n).z = 0;
    particles(n).vx = 0;
    particles(n).vy = 0;
    particles(n).vz = 0;
    particles(n).ax = 0;
    particles(n).ay = 0;
    particles(n).az = 0;
    particles(n).mass = 0; 
    
    velocityCoefficient = 1.0 ; 
    
    rand('state',1);
  
    for i = 1:n
       particles(i).x = size * rand(); 
       particles(i).y = size * rand(); 
       particles(i).z = size * rand();
       particles(i).vx = velocityCoefficient * (rand()*2 - 1); 
       particles(i).vy = velocityCoefficient * (rand()*2 - 1); 
       particles(i).vz = velocityCoefficient * (rand()*2 - 1); 
       particles(i).ax = 0; 
       particles(i).ay = 0; 
       particles(i).az = 0; 
       particles(i).mass = mass; 
    end
    
end