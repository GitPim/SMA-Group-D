#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Define the path to the directory where the data should be stored, as well as a unique identifier for each run
directory = "/data/s1968653/MWG_output/"
run_number = 10 


# In[2]:


#Here we import all the necessary dependencies
import numpy as np

from amuse.ext.orbital_elements import new_binary_from_orbital_elements, get_orbital_elements_from_binary
from amuse.ext.solarsystem import new_solar_system
from amuse.lab import units, constants, Particles, nbody_system
from amuse.io import write_set_to_file
from amuse.lab import Huayno
from amuse.couple import bridge

from tqdm import tqdm


# In[3]:


#Here we generate a galactic potential 

class MilkyWay_galaxy(object):
    def __init__(self, 
                 Mb=1.40592e10| units.MSun,
                 Md=8.5608e10| units.MSun,
                 Mh=1.07068e11 | units.MSun):
        self.Mb= Mb
        self.Md= Md
        self.Mh= Mh

    def get_potential_at_point(self,eps,x,y,z):
        r=(x**2+y**2+z**2)**0.5
        R= (x**2+y**2)**0.5
        # buldge
        b1= 0.3873 |units.kpc
        pot_bulge= -constants.G*self.Mb/(r**2+b1**2)**0.5 
        # disk
        a2= 5.31 |units.kpc
        b2= 0.25 |units.kpc
        pot_disk = -constants.G*self.Md/(R**2 + (a2+ (z**2+ b2**2)**0.5 )**2 )**0.5
        #halo
        a3= 12.0 |units.kpc
        cut_off=100 |units.kpc
        d1= r/a3
        c=1+ (cut_off/a3)**1.02
        pot_halo= -constants.G*(self.Mh/a3)*d1**1.02/(1+ d1**1.02)                   - (constants.G*self.Mh/(1.02*a3))                      * (-1.02/c +numpy.log(c) + 1.02/(1+d1**1.02)                            - numpy.log(1.0 +d1**1.02) )
        return 2*(pot_bulge+pot_disk+ pot_halo) 
                # multiply by 2 because it is a rigid potential
    
    def get_gravity_at_point(self, eps, x,y,z): 
        r= (x**2+y**2+z**2)**0.5
        R= (x**2+y**2)**0.5
        #bulge
        b1= 0.3873 |units.kpc
        force_bulge= -constants.G*self.Mb/(r**2+b1**2)**1.5 
        #disk
        a2= 5.31 |units.kpc
        b2= 0.25 |units.kpc
        d= a2+ (z**2+ b2**2)**0.5
        force_disk=-constants.G*self.Md/(R**2+ d**2 )**1.5
        #halo
        a3= 12.0 |units.kpc
        d1= r/a3
        force_halo= -constants.G*self.Mh*d1**0.02/(a3**2*(1+d1**1.02))
       
        ax= force_bulge*x + force_disk*x  + force_halo*x/r
        ay= force_bulge*y + force_disk*y  + force_halo*y/r
        az= force_bulge*z + force_disk*d*z/(z**2 + b2**2)**0.5 + force_halo*z/r 

        return ax,ay,az
    
MW_potential = MilkyWay_galaxy()


# In[4]:


#Takes a primary- and a secondary-particle and then returns the orbital parameters of their orbit. This function then returns
#the semi major axis. 

def sma_determinator(primary, secondary):
    binary = Particles(0)
    binary.add_particle(primary)
    binary.add_particle(secondary)
        
    orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
    semi_major_axis = orbital_params[2]
    return semi_major_axis


# In[5]:


#Function to generate orbits for comets in the Solar System.
def comet_positions_and_velocities(N_objects, sun_location):  
    positions = np.zeros((N_objects, 3)) | units.AU
    velocities = np.zeros((N_objects,3)) | units.kms
    
    m_sun = 1 | units.MSun
    m_comet = 0 | units.MSun #Take massless test particles
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


# In[6]:


#Function to create a post-tack system, i.e. with the giants at orbits as they are predicted to have been after the Grand Tack
def create_post_tack_giants_system():
    #Create the present day solar system and keep only the sun and the giants
    present_day_solar_system = new_solar_system()
    present_day_solar_system = present_day_solar_system[present_day_solar_system.mass > 10**-5 | units.MSun] # Takes gas giants and Sun only
    present_day_solar_system.move_to_center()
    
    #Create a post_tack_giants_system by first recreating the sun.
    post_tack_giants_system = Particles(1) 
    post_tack_giants_system[0].name = "Sun"
    post_tack_giants_system[0].mass = 1.0 | units.MSun
    post_tack_giants_system[0].radius = 1.0 | units.RSun  
    post_tack_giants_system[0].position = (0, 0, 0) | units.AU
    post_tack_giants_system[0].velocity = (0, 0, 0) | units.kms
    
    #The post tack orbital elements for the planets as below
    a =  np.array([5.4, 7.1, 10.5, 13]) | units.AU 
    true_anomalies = np.random.uniform(0, 360, 4) | units.deg
    long_of_ascending_node = np.random.uniform(0, 360, 4) | units.deg
    args_of_periapsis = np.random.uniform(0, 360, 4) | units.deg
    
    #Create the four planets as binaries with the sun and add them to the post_tack_giants_system
    for i in range(4):
        orbital_elements = get_orbital_elements_from_binary(present_day_solar_system[0]+ present_day_solar_system[i+1], G=constants.G)
        inclination = orbital_elements[5] #Make sure we have a sensable inclination for the giants
        
        
        sun_and_planet = new_binary_from_orbital_elements(post_tack_giants_system.mass[0], present_day_solar_system[i+1].mass, 
                                          a[i], 0, true_anomalies[i], inclination, long_of_ascending_node[i], args_of_periapsis[i], G=constants.G)
        
        planet = Particles(1)
        planet.name = present_day_solar_system[i+1].name
        planet.mass = present_day_solar_system[i+1].mass
        planet.radius = present_day_solar_system[i+1].radius
        planet.position = (sun_and_planet[1].x-sun_and_planet[0].x, sun_and_planet[1].y-sun_and_planet[0].y, sun_and_planet[1].z-sun_and_planet[0].z)
        planet.velocity = (sun_and_planet[1].vx-sun_and_planet[0].vx, sun_and_planet[1].vy-sun_and_planet[0].vy, sun_and_planet[1].vz-sun_and_planet[0].vz)
        post_tack_giants_system.add_particle(planet) 
        
    return post_tack_giants_system
        
post_tack_giants_system = create_post_tack_giants_system()
post_tack_giants_system.move_to_center()


# In[7]:


#Define the number of comets and create their velocities and positions
N_objects = 1*10**2
sun_location = [post_tack_giants_system[0].x.in_(units.AU), post_tack_giants_system[0].y.in_(units.AU), post_tack_giants_system[0].z.in_(units.AU)]
comet_positions, comet_velocities = comet_positions_and_velocities(N_objects, sun_location)


# In[8]:


# Here we add the comets, where orbit parameters were chosen from a uniform distribution
def add_comet_objects(post_tack_giants_system, N_objects, comet_positions, comet_velocities):
    for i in tqdm(range(N_objects)):
        comet = Particles(1)
        comet.name = "COMET_" + str(i)
        comet.mass = 0.0 | units.MSun #Take massless test particles
        comet.radius = 0.0 | units.RSun 
        comet.position = (comet_positions[i, 0], comet_positions[i, 1], comet_positions[i, 2])
        comet.velocity = (comet_velocities[i, 0], comet_velocities[i, 1], comet_velocities[i, 2])
        post_tack_giants_system.add_particle(comet)
    
    z_comp = np.arctan(100/8500.) #Determining the z-component of the sun's trajectory around the galactic center
    
    for i in range(len(post_tack_giants_system)): #adding the sun's trajectory around the galactic center
        post_tack_giants_system[i].position += (1, 0, 0) * (8.5 | units.kpc) 
        post_tack_giants_system[i].velocity += (0,np.sqrt(1-z_comp**2),z_comp) * (220 | units.kms) 
    
    
    return post_tack_giants_system

complete_post_tack_system = add_comet_objects(post_tack_giants_system, N_objects, comet_positions, comet_velocities)


# In[9]:


#Here we create the converter
converter_length = get_orbital_elements_from_binary(complete_post_tack_system[0:2], G = constants.G)[2].in_(units.AU) # Typical distance used for calculation (=distance from Sun to Jupiter)
converter=nbody_system.nbody_to_si(complete_post_tack_system.mass.sum(), 
                                   converter_length)


# In[10]:


#Here we evolve the complete_post_tack_system, without a grandtack happening or a Milky way potential being present

def MWG_evolver(complete_post_tack_system, converter, N_objects, potential, end_time, time_step):
    #Initialise the gravity code and add the particles to it
    gravity_code = Huayno(converter)
    gravity_code.particles.add_particles(complete_post_tack_system)
    channel = gravity_code.particles.new_channel_to(complete_post_tack_system)
    
    gravity_bridge = 0
    gravity_bridge = bridge.Bridge(use_threading=False)
    gravity_bridge.add_system(gravity_code, (potential,))
    gravity_bridge.timestep = 100 |units.yr
    
    times = np.arange(0., end_time, time_step) | units.yr #All time steps to which we want to evolve the model
    
    #---------------------------------------------------------------------------------------------------------
    #Here we define the planetary orbital parameters that should be returned to if the planets start moving too much 
    current_sma = np.array([0, 0, 0, 0]) | units.AU
    correct_sma =  np.array([5.4, 7.1, 10.5, 13]) | units.AU
    inclinations =  np.array([0, 0, 0, 0]) | units.deg
    
    
    system = new_solar_system()
    system = system[system.mass > 10**-5 | units.MSun] # Takes gas giants and Sun only
    system.move_to_center()
    for k in range(4):
        orbital_elements = get_orbital_elements_from_binary(system[0]+ system[k+1], G=constants.G)
        inclinations[k] =  orbital_elements[5]
    #------------------------------------------------------------------------------------------------------------
    dead_comets = [] #Here all 'dead' comets are stored
    
    #Below the evolving starts
    for i in tqdm(range(len(times))):
        gravity_bridge.evolve_model(times[i])
        channel.copy()
        
        #---------------------------------------------------------------------------------------------------------------
        #Here we check if the planetary orbits are still 'correct' and act for three degrees of incorrectness.
        for j in range(4):
            current_sma[j] = sma_determinator(gravity_code.particles[0], gravity_code.particles[j+1])
        
        for l in range(4):
            if abs(current_sma[l]/correct_sma[l]) > 1.25 or abs(current_sma[l]/correct_sma[l]) < 0.75: #The orbits are too much perturbed, so we end the simulation
                return
        
            elif abs(current_sma[l]/correct_sma[l]) > 1.05 or abs(current_sma[l]/correct_sma[l]) < 0.95: #The orbits are slightly perturbed, so we redefinie them
                print("Here", complete_post_tack_system[l+1].name, "was redefined")
                binary = Particles(0)
                binary.add_particle(gravity_code.particles[0])
                binary.add_particle(gravity_code.particles[l+1])

                orbital_params = get_orbital_elements_from_binary(binary, G = constants.G)
                true_anomaly, ascending_node, pericenter = orbital_params[4].in_(units.deg), orbital_params[6].in_(units.deg), orbital_params[7].in_(units.deg)

                sun_and_plan = new_binary_from_orbital_elements(1 | units.MSun, orbital_params[1], #We keep the current angles, but change the a, e and i back
                                                      correct_sma[l], 0, true_anomaly, inclinations[l], ascending_node, pericenter, G=constants.G)
                
                gravity_code.particles[l+1].position = (sun_and_plan[1].x+gravity_code.particles[0].x, sun_and_plan[1].y+gravity_code.particles[0].y, sun_and_plan[1].z+gravity_code.particles[0].z)
                gravity_code.particles[l+1].velocity = (sun_and_plan[1].vx+gravity_code.particles[0].vx, sun_and_plan[1].vy+gravity_code.particles[0].vy, sun_and_plan[1].vz+gravity_code.particles[0].vz)
            else: #The orbits do not need changing
                pass
        #----------------------------------------------------------------------------------------------------------------------
        #Once we checked for the orbital correctness, we can save data 
        
        if i%4000 == 0:
            write_set_to_file(gravity_code.particles, directory + 'MWG_run' + str(run_number) +'_time=' + str(np.log10(times[i].value_in(units.yr)))[0:5] + '.hdf5', format='hdf5', overwrite_file = True)
            
        #--------------------------------------------------------------------------------------------------------------------
        #Here we look for 'escaped' and 'out of bounds' comets
        out_of_bounds, escaped_comets = [], []
        for i in range(len(gravity_code.particles)):
            if (gravity_code.particles[i].position-gravity_code.particles[0].position).length() > 500 | units.AU:
                escaped_comets.append(gravity_code.particles[i])
                if (gravity_code.particles[i].position-gravity_code.particles[0].position).length() > 250000 | units.AU:
                    out_of_bounds.append(gravity_code.particles[i])
                    dead_comets.append(gravity_code.particles[i])
        for particle in out_of_bounds: #Out of bounds comets are removed completely
            complete_post_tack_system.remove_particle(particle)
            complete_post_tack_system.synchronize_to(gravity_code.particles)
            
        
        if i%100 == 0:
            print("The amount of currently escaped comets is ", len(escaped_comets))
            print("The amount of dead comets is ", len(dead_comets))
            for m in range(4):
                print(complete_post_tack_system[m+1].name, " is at ", (gravity_code.particles[m+1].position-gravity_code.particles[0].position).length().in_(units.AU))
    
        
    gravity_code.stop()
    write_set_to_file(gravity_code.orbiters, directory + 'MWG_run' + str(run_number) +'_final.hdf5', format='hdf5', overwrite_file = True)
    return complete_post_tack
    
    
MWG_evolved_system = MWG_evolver(complete_post_tack_system, converter, N_objects, MW_potential, end_time= 10**8, time_step= 1.25*10**3)


# In[ ]:




