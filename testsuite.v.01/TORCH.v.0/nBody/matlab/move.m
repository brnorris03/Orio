function particle = move(particle, size, dt, type)
% 	Integrate the ODE.
%     Slightly simplified Velocity Verlet integration.
%     Conserves energy better than explicit Euler method.
% 
%     Input:
%     particle_t *p        Address of particle to move.
%     int type             Force type. If using force type 1, bounce off walls.
% 
%     Output:
%     Velocity and position of current particle is updated.
%     
%  Alex Kaiser, LBNL, 10/2010   
% 
 

    %adjust velocities 
    particle.vx = particle.vx + particle.ax * dt ;
    particle.vy = particle.vy + particle.ay * dt ;
    particle.x = particle.x + particle.vx * dt ;
    particle.y = particle.y + particle.vy * dt ;

    % bounce from walls
    % only use for cutoff method 
    % account for additional size in any subsequent plots
    if (type == 1) 
        
        while( (particle.x < 0) || (particle.x > size) )
           if particle.x < 0
               particle.x = -particle.x; 
           else
               particle.x = 2*size - particle.x; 
           end
           particle.vx = -particle.vx ;
        end

        while( (particle.y < 0) || (particle.y > size) )
           if particle.y < 0
               particle.y = -particle.y; 
           else
               particle.y = 2*size - particle.y; 
           end
           particle.vy = -particle.vy ; 
        end
    
    end
    
end