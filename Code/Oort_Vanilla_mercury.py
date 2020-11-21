#!/usr/bin/env python
# coding: utf-8

# In[ ]:


directory = "/data/s1968653/Vanilla_output/"


# In[ ]:


#Here we import all the necessary dependencies
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from IPython.display import clear_output
from amuse.lab import units, constants
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import get_orbital_elements_from_binary
from amuse.ext.orbital_elements import get_orbital_elements_from_binaries
from amuse.lab import Particles
from amuse.lab import nbody_system
from amuse.couple import bridge
from amuse.lab import Rebound
from amuse.lab import Mercury
from amuse.community.ph4.interface import ph4
from amuse.io import write_set_to_file, read_set_from_file
from amuse.community.mercury.interface import MercuryWayWard
from amuse.ext.solarsystem import new_solar_system


# In[ ]:


# Function to generate random orbits for asteroids in the Solar System.
def random_positions_and_velocities(N_objects, sun_loc):  
    positions = np.zeros((N_objects, 3)) | units.AU
    velocities = np.zeros((N_objects,3)) | units.kms
    
    m_sun = 1 | units.MSun
    m_oort = 0 | units.MSun
    for i in range(N_objects):
        # Values below correspond with random locations anywhere in the Solar System
        a = np.random.uniform(4, 40) | units.AU  # semi-major axis
        e = np.random.uniform(0, 0.05)  # eccentricity
        inclination = np.random.uniform(-5, 5) | units.deg
        true_anomaly = np.random.uniform (0, 360) | units.deg
        arg_of_periapsis = np.random.uniform(0, 360) | units.deg
        long_of_ascending_node = np.random.uniform(0, 360) | units.deg
        sun_and_oort = new_binary_from_orbital_elements(m_sun, m_oort, 
                                          a, e, true_anomaly, inclination, long_of_ascending_node, arg_of_periapsis, G=constants.G)
        positions[i] = (sun_and_oort[1].x+sun_loc[0]), (sun_and_oort[1].y+sun_loc[1]), (sun_and_oort[1].z+sun_loc[2])
        velocities[i]= sun_and_oort[1].vx, sun_and_oort[1].vy, sun_and_oort[1].vz
    return positions, velocities


# In[ ]:


def merge_two_bodies(bodies, particles_in_encounter):
    com_pos = particles_in_encounter.center_of_mass()
    com_vel = particles_in_encounter.center_of_mass_velocity()
    d = (particles_in_encounter[0].position - particles_in_encounter[1].position)
    v = (particles_in_encounter[0].velocity - particles_in_encounter[1].velocity)
    print("Actually merger occurred:")
    print("Two objects (M=",particles_in_encounter.mass.in_(units.MSun),
          ") collided with d=", d.length().in_(units.au))
    #time.sleep(10)
    new_particle=Particles(1)
    new_particle.mass = particles_in_encounter.total_mass()
    new_particle.position = com_pos
    new_particle.velocity = com_vel
    new_particle.radius = ((particles_in_encounter.radius**3).sum())**(1./3.) # New radius cannot simply be summed
    bodies.add_particles(new_particle)
    bodies.remove_particles(particles_in_encounter)


# In[ ]:


# To resolve a collision between two or more particles, we merge them together using their center of mass (velocity).
def resolve_collision(collision_detection, gravity_code, bodies, time):
    print("Well, we have an actual collision between two or more objects.")
    print("This happened at time=", time.in_(units.yr))
    for ci in range(len(collision_detection.particles(0))):  # Multiple pairs of collisions could take place in one timestep
        encountering_particles = Particles(particles=[collision_detection.particles(0)[ci],
                                                          collision_detection.particles(1)[ci]])
        colliding_objects = encountering_particles.get_intersecting_subset_in(bodies)
        merge_two_bodies(bodies, colliding_objects)
        bodies.synchronize_to(gravity_code.particles) # Update bodies to contain 1 particle instead of 2
        
def sma_determinator(primary, secondary):
    binary = Particles(0)
    binary.add_particle(primary)
    binary.add_particle(secondary)
        
    orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
    return orbital_params[2]


# In[ ]:


def create_system():
    system = new_solar_system()
    system = system[system.mass > 10**-5 | units.MSun] # Takes gas giants and Sun only
    system.move_to_center()
    
    sun = Particles(1)
    sun.name = "Sun"
    sun.mass = 1.0 | units.MSun
    sun.radius = 1.0 | units.RSun  
    sun.position = (0, 0, 0) | units.AU
    sun.velocity = (0, 0, 0) | units.kms
    sun.density = 3*sun.mass/(4*np.pi*sun.radius**3)
    
    names = ["Jupiter", "Saturn", "Uranus", "Neptune"]
    masses = [317.8, 90, 15, 17] | units.MEarth
    radii = [0.10049, 0.083703, 0.036455, 0.035392] | units.RSun
    semis = [5.4, 7.1, 10.5, 13] | units.AU #[3.5, 4.9, 6.4, 8.4]
    trues = [0, 90, 180, 270]| units.deg#np.random.uniform(0, 360, 4) | units.deg
    longs = np.random.uniform(0, 360, 4) | units.deg
    args = np.random.uniform(0, 360, 4) | units.deg
    
    for i in range(4):
        orbital_elements = get_orbital_elements_from_binary(system[0]+ system[i+1], G=constants.G)
        eccentricity, inclination = 0, orbital_elements[5]
        
        
        sun_and_plan = new_binary_from_orbital_elements(sun[0].mass, masses[i], 
                                          semis[i], 0, trues[i], inclination, longs[i], args[i], G=constants.G)
        
        planet = Particles(1)
        planet.name = system[i+1].name
        planet.mass = system[i+1].mass
        planet.radius = system[i+1].radius # This is purely non-zero for collisional purposes
        planet.position = (sun_and_plan[1].x-sun_and_plan[0].x, sun_and_plan[1].y-sun_and_plan[0].y, sun_and_plan[1].z-sun_and_plan[0].z)
        planet.velocity = (sun_and_plan[1].vx-sun_and_plan[0].vx, sun_and_plan[1].vy-sun_and_plan[0].vy, sun_and_plan[1].vz-sun_and_plan[0].vz)
        planet.density = 3*planet.mass/(4*np.pi*planet.radius**3)
        sun.add_particle(planet)
        
    return sun
        
basic_giants_system = create_system()
basic_giants_system.move_to_center()


# In[ ]:


#Define the number of asteroids and create random velocities and positions
N_objects = 1*10**2
sun_loc = [basic_giants_system[0].x.in_(units.AU), basic_giants_system[0].y.in_(units.AU), basic_giants_system[0].z.in_(units.AU)]
positions, velocities = random_positions_and_velocities(N_objects, sun_loc)


# In[ ]:


# Here we add the asteroids, where orbit parameters were chosen from a uniform distribution
def add_comet_objects(system, N_objects, rand_pos, rand_vel):
    for i in tqdm(range(N_objects)):
        oort = Particles(1)
        oort.name = "OORT_" + str(i)
        oort.mass = 0.0 | units.MSun
        oort.radius = (2.3 | units.km).in_(units.RSun) # This is purely non-zero for collisional purposes
        oort.position = (rand_pos[i, 0], rand_pos[i, 1], rand_pos[i, 2])
        oort.velocity = (rand_vel[i, 0], rand_vel[i, 1], rand_vel[i, 2])

        system.add_particle(oort)
        
    for particle in system: #So that Mercury will work
        particle.density = 3*particle.mass/(4*np.pi*particle.radius**3)
    
    return system

complete_system = add_comet_objects(basic_giants_system, N_objects, positions, velocities)


# In[ ]:


final_system = complete_system
final_system.move_to_center()


# In[ ]:


#Here we perform the conversion for the system
converter_length = get_orbital_elements_from_binary(final_system[0:2], G = constants.G)[2].in_(units.AU) # Typical distance used for calculation (=distance from Sun to Jupiter)
final_converter=nbody_system.nbody_to_si(final_system.mass.sum(), 
                                   converter_length)


# In[ ]:


#Here we evolve the basic system, without grandtack or Milky way potential

def vanilla_evolver(particle_system, converter, N_objects, end_time, time_step):
    
    names = ['Sun', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    
    gravity_code = MercuryWayWard()
    gravity_code.initialize_code()
    
    gravity_code.central_particle.add_particle(particle_system[0])
    gravity_code.orbiters.add_particles(particle_system[1:])
    gravity_code.commit_particles
    
    ch_g2l = gravity_code.particles.new_channel_to(particle_system)
    
    times = np.arange(0., end_time, time_step) | units.yr
    
    dead_comets = []
    #---------------------------------------------------------------------------------------------------------
    sma = [0, 0, 0, 0] | units.AU
    correct_sma = [5.4, 7.1, 10.5, 13] | units.AU
    eccentricities = [0, 0, 0, 0]
    inclinations = [0, 0, 0, 0] | units.deg
    
    system = new_solar_system()
    system = system[system.mass > 10**-5 | units.MSun] # Takes gas giants and Sun only
    system.move_to_center()
    for k in range(4):
        orbital_elements = get_orbital_elements_from_binary(system[0]+ system[k+1], G=constants.G)
        inclinations[k] =  orbital_elements[5]
    #------------------------------------------------------------------------------------------------------------
    
    
    
    for i in tqdm(range(len(times))):
        gravity_code.evolve_model(times[i])
        ch_g2l.copy()
        
        #---------------------------------------------------------------------------------------------------------------
        for j in range(4):
            sma[j] = sma_determinator(gravity_code.central_particle, gravity_code.orbiters[j])
        
        for l in range(4):
            if abs(sma[l]/correct_sma[l]) > 1.25 or abs(sma[l]/correct_sma[l]) < 0.75:
                return
        
            elif abs(sma[l]/correct_sma[l]) > 1.05 or abs(sma[l]/correct_sma[l]) < 0.95:
                print("Here", names[l+1], "was redefined")
                binary = Particles(0)
                binary.add_particle(gravity_code.central_particle)
                binary.add_particle(gravity_code.orbiters[l])

                orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
                true_anomaly, ascending_node, pericenter = orbital_params[4].in_(units.deg), orbital_params[6].in_(units.deg), orbital_params[7].in_(units.deg)

                sun_and_plan = new_binary_from_orbital_elements(1 | units.MSun, orbital_params[1], 
                                                      correct_sma[l], eccentricities[l], true_anomaly, inclinations[l], ascending_node, pericenter, G=constants.G)

                gravity_code.particles[l+1].position = (sun_and_plan[1].x-sun_and_plan[0].x, sun_and_plan[1].y-sun_and_plan[0].y, sun_and_plan[1].z-sun_and_plan[0].z)
                gravity_code.particles[l+1].velocity = (sun_and_plan[1].vx-sun_and_plan[0].vx, sun_and_plan[1].vy-sun_and_plan[0].vy, sun_and_plan[1].vz-sun_and_plan[0].vz)
            else:
                pass
        #----------------------------------------------------------------------------------------------------------------------
        
        
        
            
        if i%1000 == 0:
            write_set_to_file(gravity_code.orbiters, directory + 'Vanilla_run16_time=' + str(np.log10(times[i].value_in(units.yr)))[0:5] + '.hdf5', format='hdf5', overwrite_file = True)
        
        out_of_bounds, escaped_comets = [], []
        for i in range(len(particle_system)):
            if particle_system[i].position.length() > 500 | units.AU:
                escaped_comets.append(particle_system[i].name)
                if particle_system[i].position.length() > 250000 | units.AU:
                    out_of_bounds.append(particle_system[i])
                    dead_comets.append(particle_system[i])
        for particle in out_of_bounds:
            particle_system.remove_particle(particle)
            particle_system.synchronize_to(gravity_code.particles)
            
        print("The amount of currently escaped comets is ", len(escaped_comets))
        print("The amount of dead comets is ", len(dead_comets))
        print("The planetary positions are ", gravity_code.orbiters[0].position.length().in_(units.AU), gravity_code.orbiters[1].position.length().in_(units.AU), gravity_code.orbiters[2].position.length().in_(units.AU), gravity_code.orbiters[3].position.length().in_(units.AU))
    
    gravity_code.stop()
    write_set_to_file(gravity_code.orbiters, directory + 'Vanilla_run16_final.hdf5', format='hdf5', overwrite_file = True)
    return particle_system
    
    
vanilla_evolved_system = vanilla_evolver(final_system, final_converter, N_objects, end_time= 10**8, time_step= 5*10**3)

