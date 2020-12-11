#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Define the path to the directory where the data should be stored, as well as a unique identifier for each run
directory = "/data/s1968653/Vanilla_Tack_output/"
run_number = 13


# In[ ]:


#Here we import all the necessary dependencies
import numpy as np
import math

from tqdm import tqdm
from amuse.lab import units, constants, Particles, nbody_system
from amuse.ext.orbital_elements import new_binary_from_orbital_elements, get_orbital_elements_from_binary
from amuse.community.mercury.interface import MercuryWayWard
from amuse.io import write_set_to_file


# In[ ]:


#Takes a primary- and a secondary-particle and then returns the orbital parameters of their orbit. This function then returns
#the semi major axis.

def sma_determinator(primary, secondary):
    binary = Particles(0)
    binary.add_particle(primary)
    binary.add_particle(secondary)
        
    orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
    return orbital_params[2]


# In[ ]:


# Function to generate orbits for comets in the Solar System.
def comet_positions_and_velocities(N_objects, sun_location):  
    positions = np.zeros((N_objects, 3)) | units.AU
    velocities = np.zeros((N_objects,3)) | units.kms
    
    m_sun = 1 | units.MSun
    m_comet = 0 | units.MSun
    for i in range(N_objects):
        # Values below correspond with random locations anywhere in the Solar System, based of relevant literature
        a = np.random.uniform(4, 40) | units.AU  # semi-major axis
        e = np.random.uniform(0, 0.05)  # eccentricity
        inclination = np.random.uniform(-5, 5) | units.deg
        true_anomaly = np.random.uniform (0, 360) | units.deg
        arg_of_periapsis = np.random.uniform(0, 360) | units.deg
        long_of_ascending_node = np.random.uniform(0, 360) | units.deg
        sun_and_comet = new_binary_from_orbital_elements(m_sun, m_comet, 
                                          a, e, true_anomaly, inclination, long_of_ascending_node, arg_of_periapsis, G=constants.G)
        positions[i] = (sun_and_comet[1].x+sun_location[0]), (sun_and_comet[1].y+sun_location[1]), (sun_and_comet[1].z+sun_location[2])
        velocities[i]= sun_and_comet[1].vx, sun_and_comet[1].vy, sun_and_comet[1].vz
    return positions, velocities


# In[ ]:


def create_pre_tack_giants_system():
    #Create a pre_tack_giants_system by first recreating the sun.
    pre_tack_giants_system = Particles(1)
    pre_tack_giants_system[0].name = "Sun"
    pre_tack_giants_system[0].mass = 1.0 | units.MSun
    pre_tack_giants_system[0].radius = 1.0 | units.RSun  
    pre_tack_giants_system[0].position = (0, 0, 0) | units.AU
    pre_tack_giants_system[0].velocity = (0, 0, 0) | units.kms
    pre_tack_giants_system[0].density = 3*pre_tack_giants_system[0].mass/(4*np.pi*pre_tack_giants_system[0].radius**3)
    
    #The pre tack orbital elements for the planets as below
    names = ["Jupiter", "Saturn", "Uranus", "Neptune"]
    masses = np.array([317.8, 30, 5, 5]) | units.MEarth
    radii = np.array([0.10049, 0.083703, 0.036455, 0.035392]) | units.RSun
    a = np.array([3.5, 4.5, 6.013, 8.031]) | units.AU 
    inclinations = np.random.uniform(-5, 5, 4) | units.deg
    true_anomalies = np.random.uniform(0, 360, 4) | units.deg
    longs_of_ascending_node = np.random.uniform(0, 360, 4) | units.deg
    args_of_periapsis = np.random.uniform(0, 360, 4) | units.deg
    
    #Create the four planets as binaries with the sun and add them to the pre_tack_giants_system
    for i in range(4):
        sun_and_planet = new_binary_from_orbital_elements(pre_tack_giants_system[0].mass, masses[i], 
                                          a[i], 0, true_anomalies[i], inclinations[i] , longs_of_ascending_node[i], args_of_periapsis[i], G=constants.G)
        
        planet = Particles(1)
        planet.name = names[i]
        planet.mass = masses[i]
        planet.radius = radii[i] # This is purely non-zero for collisional purposes
        planet.position = (sun_and_planet[1].x-(0 | units.AU), sun_and_planet[1].y-(0 | units.AU), sun_and_planet[1].z-(0 | units.AU))
        planet.velocity = (sun_and_planet[1].vx-(0 | units.kms), sun_and_planet[1].vy-(0 | units.kms), sun_and_planet[1].vz-(0 | units.kms))
        planet.density = 3*planet.mass/(4*np.pi*planet.radius**3)
        pre_tack_giants_system.add_particle(planet)
        
    return pre_tack_giants_system
        
pre_tack_giants_system = create_pre_tack_giants_system()
pre_tack_giants_system.move_to_center()


# In[ ]:


#Define the number of asteroids and create random velocities and positions
N_objects = 1*10**2
sun_location = [pre_tack_giants_system[0].x.in_(units.AU), pre_tack_giants_system[0].y.in_(units.AU), pre_tack_giants_system[0].z.in_(units.AU)]
comet_positions, comet_velocities = comet_positions_and_velocities(N_objects, sun_location)


# In[ ]:


# Here we add the comets, where orbit parameters were chosen from a uniform distribution
def add_comet_objects(pre_tack_giants_system, N_objects, comet_positions, comet_velocities):
    for i in tqdm(range(N_objects)):
        comet = Particles(1)
        comet.name = "OORT_" + str(i)
        comet.mass = 0.0 | units.MSun #Take massless test particles
        comet.radius = 0.0 | units.RSun
        comet.position = (comet_positions[i, 0], comet_positions[i, 1], comet_positions[i, 2])
        comet.velocity = (comet_velocities[i, 0], comet_velocities[i, 1], comet_velocities[i, 2])
        comet.density = 0.0 | (units.MSun/(units.RSun**3))

        pre_tack_giants_system.add_particle(comet)
    return pre_tack_giants_system

complete_pre_tack_system = add_comet_objects(pre_tack_giants_system, N_objects, comet_positions, comet_velocities)


# In[ ]:


#Here we create the conditions for the migration of the planets.
#There are three parts to the migration: Jupiter moving in, Saturn moving in,
#and everything moving out. For each part, we determine the value that the
#semi major axis of a planet should have in the next timestep. In the evolver,
#we then redetermine the orbital parameters for a planet with this new semi
#major axis. 

#Each part is written so that the migration starts at semi major axis a_start,
#and ends at a_end, in a time of time_scale

def semi_major_axis_next_step_in_jup(time_now, time_scale, a_start, a_end):
    travel_distance = a_start-a_end
    sma_next_step = a_start - travel_distance*(1/(1-1/math.e))*(1-np.exp(-(time_now)/time_scale))
    return sma_next_step
    
def semi_major_axis_next_step_in_sat(time_now, time_scale, a_start, a_end):
    travel_distance = a_start-a_end
    sma_next_step = a_start - travel_distance*(1/(1-1/math.e))*(1-np.exp(-(time_now-(10**5 | units.yr))/time_scale))
    return sma_next_step

def semi_major_axis_next_step_out(time_now, time_start, a_end, a_start, time_scale):    
    travel_distance = a_end-a_start
    sma_next_step = a_start + travel_distance*(1/(1-1/math.e))*(1-np.exp(-(time_now-time_start)/time_scale))
    return sma_next_step


# In[ ]:


#Here we create the converter
converter_length = get_orbital_elements_from_binary(complete_pre_tack_system[0:2], G = constants.G)[2].in_(units.AU) # Typical distance used for calculation (=distance from Sun to Jupiter)
converter=nbody_system.nbody_to_si(complete_pre_tack_system.mass.sum(), 
                                   converter_length)


# In[ ]:


#Here we manually create all the timesteps that we want the model to evolve to
#Due to the different timescales of migration, each stage in the migration requires
#It's own time array. We also manually create the times at which to save the data files

jupiter_inward_times = np.arange(0, 1*10**5, 3) 
saturn_inward_times = np.arange(1*10**5, 1.025*10**5, 0.5) 
outward_times = np.arange(1.025*10**5, 6*10**5, 15) 
post_tack_times = np.arange(6*10**5, 10**8, 1*10**3) 

final_time_range = np.concatenate((jupiter_inward_times, saturn_inward_times, outward_times, post_tack_times)) | units.yr
save_file_times = np.concatenate((jupiter_inward_times[::int(len(jupiter_inward_times)/5)], 
                                  saturn_inward_times[::int(len(saturn_inward_times)/5)], 
                                  outward_times[::int(len(outward_times)/5)], 
                                  post_tack_times[::int(len(post_tack_times)/20)])) | units.yr


# In[ ]:


def vanilla_tack_evolver(complete_pre_tack_system, converter, N_objects, times, save_file_times):
    #Initialise the gravity code and add the particles to it
    
    gravity_code = MercuryWayWard(converter)
    gravity_code.initialize_code()
    
    gravity_code.central_particle.add_particle(complete_pre_tack_system[0])
    gravity_code.orbiters.add_particles(complete_pre_tack_system[1:])
    gravity_code.commit_particles
    
    channel = gravity_code.particles.new_channel_to(complete_pre_tack_system)
    
    #----------------------------------------------------------------------------------------------------
    #Here we define the 'correct' sma's for the different migrations. Also, the initial
    #planetary inclinations are stored for later use.
    
    initial_sma = np.array([3.5, 4.5, 6.013, 8.031]) | units.AU
    saturn_sma = np.array([1.5, 4.5, 6.013, 8.031]) | units.AU
    outward_sma = np.array([1.5, 1.5*((3/2)**(2/3)), 6.013, 8.031]) | units.AU
    post_tack_sma = np.array([5.4, 7.1, 10.5, 13.]) | units.AU
    current_sma = [3.5, 4.5, 6.013, 8.031] | units.AU
    
    inclinations = [0, 0, 0, 0] | units.deg
    
    for k in range(4):
        orbital_elements = get_orbital_elements_from_binary(complete_pre_tack_system[0]+ complete_pre_tack_system[k+1], G=constants.G)
        inclinations[k] =  orbital_elements[5]
    #----------------------------------------------------------------------------------------------------
    #Here we define the parameters used in the 'semi_major_axis_next_step' functions
    #The sma's are based on exact resonances. The time_scales are taken from literature.
    #The 0 values will be changed during the evolution of the model.
    #pre_resonant is used in the outward migration, when jupiter and saturn are
    #already in resonance with eachother, so that pre_resonant = True
    
    a_start = [1.5, 1.5*((3/2)**(2/3)), 0, 0] | units.AU
    a_end = [5.4, 5.4*((3/2)**(2/3)), 5.4*((3/2)**(2/3)*(9/5)**(2/3)), 5.4*((3/2)**(2/3)*(5/2)**(2/3))] | units.AU
    time_start = [1.025*10**5, 1.025*10**5, 0, 0] | units.yr
    time_scale = [5*10**5, 5*10**5, 0, 0] | units.yr
    
    resonances = [2/3, 3/2, 9/5, 5/2]
    pre_resonant = [False, False, True, True]
    
    outward_migration_started = False
    
    dead_comets = []
    
    
    
    #----------------------------------------------------------------------------------------------------
    #Below, the evolution starts.
    
    for i in tqdm(range(len(times)-1)):
        gravity_code.evolve_model(times[i])
        channel.copy()
        
        #Save the model when we want it to
        if times[i] in save_file_times:
            write_set_to_file(gravity_code.orbiters, directory + 'Vanilla_Tack_run' + str(run_number)+ '_time=' + str(np.log10(times[i].value_in(units.yr)))[0:5] + '.hdf5', format='hdf5', overwrite_file = True)
        
        #For each timestep determine the current sma's
        for j in range(4):
            current_sma[j] = sma_determinator(gravity_code.central_particle, gravity_code.orbiters[j])

        #-----------------------------------------------------------------------------------------------------
        #This chunk of code describes the inward migration of jupiter
        #The first orbiter and the second particle in the gravity_code. 
        if times[i] < 10**5 | units.yr :
            #This pushes Jupiter slightly inward
            sma_next_step = semi_major_axis_next_step_in_jup(times[i+1], 10**5 | units.yr, 3.5 | units.AU, 1.5 | units.AU)
            
            binary = Particles(0)
            binary.add_particle(gravity_code.central_particle)
            binary.add_particle(gravity_code.orbiters[0])

            orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
            true_anomaly, ascending_node, pericenter = orbital_params[4].in_(units.deg), orbital_params[6].in_(units.deg), orbital_params[7].in_(units.deg)

            sun_and_planet = new_binary_from_orbital_elements(1 | units.MSun, orbital_params[1], 
                                              sma_next_step, 0, true_anomaly, inclinations[0], ascending_node, pericenter, G=constants.G)

            gravity_code.particles[1].position = (sun_and_planet[1].x-(0 | units.AU), sun_and_planet[1].y-(0 | units.AU), sun_and_planet[1].z-(0 | units.AU))
            gravity_code.particles[1].velocity = (sun_and_planet[1].vx-(0 | units.kms), sun_and_planet[1].vy-(0 | units.kms), sun_and_planet[1].vz-(0 | units.kms))
            
            #During the tack, the masses of the planets increase towards their current values
            gravity_code.particles[2].mass *= 2**(1.5/(10**5))
            gravity_code.particles[3].mass *= 1.2**(1.5/(10**5))
            gravity_code.particles[4].mass *= 1.2**(1.5/(10**5))
            
            #This keeps the other planets in place
            for j in range(3):
                binary = Particles(0)
                binary.add_particle(gravity_code.central_particle)
                binary.add_particle(gravity_code.orbiters[j+1])

                orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
                true_anomaly, ascending_node, pericenter = orbital_params[4].in_(units.deg), orbital_params[6].in_(units.deg), orbital_params[7].in_(units.deg)

                sun_and_planet = new_binary_from_orbital_elements(1 | units.MSun, orbital_params[1], 
                                              initial_sma[1+j], 0, true_anomaly, inclinations[j+1], ascending_node, pericenter, G=constants.G)

                gravity_code.particles[j+2].position = (sun_and_planet[1].x-(0 | units.AU), sun_and_planet[1].y-(0 | units.AU), sun_and_planet[1].z-(0 | units.AU))
                gravity_code.particles[j+2].velocity = (sun_and_planet[1].vx-(0 | units.kms), sun_and_planet[1].vy-(0 | units.kms), sun_and_planet[1].vz-(0 | units.kms))

        #------------------------------------------------------------------------------------------------------------
        #This chunk of code describes the inward migration of saturn
        elif 1*10**5 | units.yr <= times[i] < 1.025*10**5 | units.yr:
            #This pushes Saturn slightly inward
            sma_next_step = semi_major_axis_next_step_in_sat(times[i+1], 2.5*10**3 | units.yr, 4.5 | units.AU, 1.5*((3/2)**(2/3)) | units.AU)
            
            binary = Particles(0)
            binary.add_particle(gravity_code.central_particle)
            binary.add_particle(gravity_code.orbiters[1])

            orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
            true_anomaly, ascending_node, pericenter = orbital_params[4].in_(units.deg), orbital_params[6].in_(units.deg), orbital_params[7].in_(units.deg)

            sun_and_planet = new_binary_from_orbital_elements(1 | units.MSun, orbital_params[1], 
                                              sma_next_step, 0, true_anomaly, inclinations[1], ascending_node, pericenter, G=constants.G)

            gravity_code.particles[2].position = (sun_and_planet[1].x-(0 | units.AU), sun_and_planet[1].y-(0 | units.AU), sun_and_planet[1].z-(0 | units.AU))
            gravity_code.particles[2].velocity = (sun_and_planet[1].vx-(0 | units.kms), sun_and_planet[1].vy-(0 | units.kms), sun_and_planet[1].vz-(0 | units.kms))
            
            gravity_code.particles[2].mass *= 1.5**(0.5/(2500))
            gravity_code.particles[3].mass *= (17.15/6)**(0.5/(5*10**5))
            gravity_code.particles[4].mass *= (14.54/6)**(0.5/(5*10**5))
            
            #This keeps the other planets in place
            for j in [0, 2, 3]:
                binary = Particles(0)
                binary.add_particle(gravity_code.central_particle)
                binary.add_particle(gravity_code.orbiters[j])

                orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
                true_anomaly, ascending_node, pericenter = orbital_params[4].in_(units.deg), orbital_params[6].in_(units.deg), orbital_params[7].in_(units.deg)

                sun_and_planet = new_binary_from_orbital_elements(1 | units.MSun, orbital_params[1], 
                                              saturn_sma[j], 0, true_anomaly, inclinations[j], ascending_node, pericenter, G=constants.G)

                gravity_code.particles[j+1].position = (sun_and_planet[1].x-(0 | units.AU), sun_and_planet[1].y-(0 | units.AU), sun_and_planet[1].z-(0 | units.AU))
                gravity_code.particles[j+1].velocity = (sun_and_planet[1].vx-(0 | units.kms), sun_and_planet[1].vy-(0 | units.kms), sun_and_planet[1].vz-(0 | units.kms))

        #------------------------------------------------------------------------------------------------------------    
        #This chunk of code describes the outward migration of all planets
        elif 1.025*10**5 | units.yr <= times[i] < 6*10**5 | units.yr:
                
            #This bit checks if uranus and neptune already should start migrating
            for k in range(4):
                if pre_resonant[k] == True:
                    if current_sma[k]/current_sma[1] < (resonances[k])**(2/3):
                        pre_resonant[k] = False
                        a_start[k] = current_sma[k]
                        time_start[k] = times[i]
                        time_scale[k] = (6*10**5 | units.yr)-times[i]
            
            #If pre_resonant == False, pushes the planet outward. If true, keeps it in place
            for l in range(4):
                if pre_resonant[l] == False: 
                    sma_next_step = semi_major_axis_next_step_out(times[i+1], time_start[l], a_end[l], a_start[l], time_scale[l])
                    
                    binary = Particles(0)
                    binary.add_particle(gravity_code.central_particle)
                    binary.add_particle(gravity_code.orbiters[l])

                    orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
                    true_anomaly, ascending_node, pericenter = orbital_params[4].in_(units.deg), orbital_params[6].in_(units.deg), orbital_params[7].in_(units.deg)

                    sun_and_planet = new_binary_from_orbital_elements(1 | units.MSun, orbital_params[1], 
                                                      sma_next_step, 0, true_anomaly, inclinations[l], ascending_node, pericenter, G=constants.G)

                    gravity_code.particles[l+1].position = (sun_and_planet[1].x-(0 | units.AU), sun_and_planet[1].y-(0 | units.AU), sun_and_planet[1].z-(0 | units.AU))
                    gravity_code.particles[l+1].velocity = (sun_and_planet[1].vx-(0 | units.kms), sun_and_planet[1].vy-(0 | units.kms), sun_and_planet[1].vz-(0 | units.kms))
                               
                else:
                    binary = Particles(0)
                    binary.add_particle(gravity_code.central_particle)
                    binary.add_particle(gravity_code.orbiters[l])

                    orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
                    true_anomaly, ascending_node, pericenter = orbital_params[4].in_(units.deg), orbital_params[6].in_(units.deg), orbital_params[7].in_(units.deg)

                    sun_and_planet = new_binary_from_orbital_elements(1 | units.MSun, orbital_params[1], 
                                                      outward_sma[l], 0, true_anomaly, inclinations[l], ascending_node, pericenter, G=constants.G)

                    gravity_code.particles[l+1].position = (sun_and_planet[1].x-(0 | units.AU) , sun_and_planet[1].y-(0 | units.AU), sun_and_planet[1].z-(0 | units.AU))
                    gravity_code.particles[l+1].velocity = (sun_and_planet[1].vx-(0 | units.kms), sun_and_planet[1].vy-(0 | units.kms), sun_and_planet[1].vz-(0 | units.kms))

            gravity_code.particles[3].mass *= (17.15/6)**(15/(5*10**5))
            gravity_code.particles[4].mass *= (14.54/6)**(15/(5*10**5))
            
        #------------------------------------------------------------------------------------------------------------
        
        else:
          #---------------------------------------------------------------------------------------------------------------
          
            for l in range(4):
                if abs(current_sma[l]/post_tack_sma[l]) > 1.25 or abs(current_sma[l]/post_tack_sma[l]) < 0.75: #The orbits are too much perturbed, so we end the simulation
                    return
          
                elif abs(current_sma[l]/post_tack_sma[l]) > 1.05 or abs(current_sma[l]/post_tack_sma[l]) < 0.95: #The orbits are slightly perturbed, so we redefinie them
                    print("Here", complete_pre_tack_system[l+1].name, "was redefined")
                    binary = Particles(0)
                    binary.add_particle(gravity_code.central_particle)
                    binary.add_particle(gravity_code.orbiters[l])
  
                    orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
                    true_anomaly, ascending_node, pericenter = orbital_params[4].in_(units.deg), orbital_params[6].in_(units.deg), orbital_params[7].in_(units.deg)
  
                    sun_and_planet = new_binary_from_orbital_elements(1 | units.MSun, orbital_params[1], 
                                                        post_tack_sma[l], 0, true_anomaly, inclinations[l], ascending_node, pericenter, G=constants.G)
  
                    gravity_code.particles[l+1].position = (sun_and_planet[1].x-(0 | units.AU), sun_and_planet[1].y-(0 | units.AU), sun_and_planet[1].z-(0 | units.AU))
                    gravity_code.particles[l+1].velocity = (sun_and_planet[1].vx-(0 | units.kms), sun_and_planet[1].vy-(0 | units.kms), sun_and_planet[1].vz-(0 | units.kms))
                else:
                    pass
        #----------------------------------------------------------------------------------------------------------------------
        #Here we look for 'escaped' and 'out of bounds' comets
            out_of_bounds, escaped_comets = [], []
            for i in range(len(gravity_code.orbiters)):
                if gravity_code.orbiters[i].position.length() > 500 | units.AU:
                    escaped_comets.append(gravity_code.orbiters[i])
                if gravity_code.orbiters[i].position.length() > 250000 | units.AU:
                    out_of_bounds.append(gravity_code.orbiters[i])
                    dead_comets.append(gravity_code.orbiters[i])
            for particle in out_of_bounds:
                complete_pre_tack_system.remove_particle(particle)
                complete_pre_tack_system.synchronize_to(gravity_code.particles)
            if i%1000 == 0:
                print("The amount of currently escaped comets is ", len(escaped_comets))
                print("The amount of dead comets is ", len(dead_comets))
        
        if i%1000 == 0:
            print("The sma's are: ", current_sma[0], current_sma[1], current_sma[2], current_sma[3])
    
    gravity_code.stop()
    return complete_pre_tack_system
    

vanilla_tack_evolved_system = vanilla_tack_evolver(complete_pre_tack_system, converter, N_objects, final_time_range, save_file_times)

