function particle = applyForce3d(particle, neighbor, cutoff, min_r, type)
%
%  Applies force to a particle according to a neighboring particle.
%  Changing the force drastically changes results.
% 
%  Input:
%  particle_t *particle      Current particle.
%  particle_t *neighbor      Its neighbor to interacte with.
%  int type                  Parameter to control type of force.
%                                Type 0 = gravity-like attractive force.
%                                Optionally, this force cuts off at a small radius to keep forces from exploding at short range.
%                                This is default behavior.
%                                Type 1 = repulsive.
% 
%  Output:
%  Acceleration of current particle is updated.
% 
% 
%  Alex Kaiser, LBNL, 10/2010
% 
  

    % gravity type force. 
    if( type ~= 1)
        dx = neighbor.x - particle.x; 
        dy = neighbor.y - particle.y; 
        dz = neighbor.z - particle.z;

        G = 9.8; 
        distance = sqrt(dx^2 + dy^2 + dz^2);

        if distance <= .001
            %disp('particles close. weird results could occur. using distance cutoff'); 
            force = (G * particle.mass * neighbor.mass) / .001^3 ; 
            particle.ax = particle.ax + dx * force; 
            particle.ay = particle.ay + dy * force;
            particle.az = particle.az + dz * force;
        else
            force = (G * particle.mass * neighbor.mass) / distance^3 ; 
            particle.ax = particle.ax + dx * force; 
            particle.ay = particle.ay + dy * force;
            particle.az = particle.az + dz * force;
        end
    
    elseif (type == 1)
        dx = neighbor.x - particle.x;
        dy = neighbor.y - particle.y;
        dz = neighbor.z - particle.z;
        r2 = dx^2 + dy^2 + dz^2; 

        if r2 > cutoff^2
            return; 
        end

        r2 = max( r2, min_r^2 ); 
        r = sqrt(r2); 

        coef = (1-cutoff/r) / r2 / particle.mass ; 
        %as written all have same mass
        particle.ax = particle.ax + coef * dx; 
        particle.ay = particle.ay + coef * dy;
        particle.az = particle.az + coef * dy;
    end
    
    
end