"""
   example code for bridging a gravity solver with a hydrodynamics solver
"""
import numpy
import math
from amuse.lab import *
from amuse.couple import bridge
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary

converter = nbody_system.nbody_to_si(1|units.MSun, 1|units.au)

class TidalFriction(bridge.GravityCodeInField):
    def kick_with_field_code(self, particles, stars, dt):
        C = 1.0
        ax, ay, az = self.calculate_additional_forces(particles.position, particles.velocity, C, particles.mass)
        self.update_velocities(particles, dt, ax, ay, az)
    def drift(self, tend): 
        pass

    def calculate_additional_forces(self, pos, vel, C, masses):
        # ext_acc: the additional accelerations
        # put the routine of calculating additional accelerations here
        # for (size_t i = 0; i < 3 * N; i++) acc[i] -= 0.0; // replace this line
        if (C > 0.0):
            ax, ay, az = self.calculate_post_newtonian(pos, vel, C, masses)
        return ax, ay, az

    # well, this is all going to be in in N-body units, G=1
    def calculate_post_newtonian(self, pos, vel, C, masses):

        m = 1.0
        self.CONST_C_LIGHT_PM2 = (C/m)**2
        self.CONST_C_LIGHT_PM5 = (C/m)**5
        self.G = 1
        c8div5 = C**(8/5.)
        c17div3 = C**(17/3.)
        
        accx = [] | units.m/units.s**2
        accy = [] | units.m/units.s**2
        accz = [] | units.m/units.s**2
        for j in range(len(masses)):
            x = converter.to_nbody(pos[j].x).number
            y = converter.to_nbody(pos[j].y).number
            z = converter.to_nbody(pos[j].z).number
            vx = converter.to_nbody(vel[j][0]).number
            vy = converter.to_nbody(vel[j][1]).number
            vz = converter.to_nbody(vel[j][2]).number
            M_j = converter.to_nbody(masses[j]).number

                   
            accx.append(0|units.m/units.s**2)
            accy.append(0|units.m/units.s**2)
            accz.append(0|units.m/units.s**2)
            ax = 0 
            ay = 0 
            az = 0 
            for k in range(len(masses)):
                if (j == k):
                    break
                xk = converter.to_nbody(pos[k].x).number
                yk = converter.to_nbody(pos[k].y).number
                zk = converter.to_nbody(pos[k].z).number
                vxk = converter.to_nbody(vel[k][0]).number
                vyk = converter.to_nbody(vel[k][1]).number
                vzk = converter.to_nbody(vel[k][2]).number
                M_k = converter.to_nbody(masses[k]).number
                GM = self.G * M_k
                GM_C_PM2 = GM * self.CONST_C_LIGHT_PM2
                GM_P2_C_PM2 = GM * GM_C_PM2

                dx = x - xk
                dy = y - yk
                dz = z - zk
                dvx = vx - vxk
                dvy = vy - vyk
                dvz = vz - vzk

                r_dot_v = dx * dvx + dy * dvy + dz * dvz
                v_dot_v = dvx * dvx + dvy * dvy + dvz * dvz

                rel_sep2 = dx * dx + dy * dy + dz * dz
                rel_sep1 = numpy.sqrt(rel_sep2)

                rel_sep_div1 = 1.0 / rel_sep1
                rel_sep_div2 = rel_sep_div1 * rel_sep_div1
                rel_sep_div3 = rel_sep_div1 * rel_sep_div2
                rel_sep_div4 = rel_sep_div1 * rel_sep_div3

                # 1PN //
                FAC0_1PN = 4.0 * GM_C_PM2 * r_dot_v * rel_sep_div3
                FAC1_1PN = 4.0 * GM_P2_C_PM2 * rel_sep_div4
                FAC2_1PN = -1*GM_C_PM2 * v_dot_v * rel_sep_div3

                ax += FAC0_1PN * dvx + FAC1_1PN * dx + FAC2_1PN * dx
                ay += FAC0_1PN * dvy + FAC1_1PN * dy + FAC2_1PN * dy
                az += FAC0_1PN * dvz + FAC1_1PN * dz + FAC2_1PN * dz

                #// 2PN //
                GM_P2_RC_PM4 = GM_P2_C_PM2 * self.CONST_C_LIGHT_PM2 * rel_sep_div4
                FAC0_2PN = 2.0 * r_dot_v * r_dot_v * rel_sep_div2
                FAC1_2PN = -9.0 * GM * rel_sep_div1
                FAC2_2PN = -2.0 * r_dot_v

                ax += GM_P2_RC_PM4*(FAC0_2PN * dx + FAC1_2PN * dx + FAC2_2PN * dvx)
                ay += GM_P2_RC_PM4*(FAC0_2PN * dy + FAC1_2PN * dy + FAC2_2PN * dvy)
                az += GM_P2_RC_PM4*(FAC0_2PN * dz + FAC1_2PN * dz + FAC2_2PN * dvz)

                #// 2.5PN //
                M_tot = M_j + M_k
                nu = M_j * M_k / (M_tot * M_tot)
                GM_tot = self.G * M_tot
                FAC0_25PN = -c8div5 * GM_tot * GM_tot * nu * self.CONST_C_LIGHT_PM5 * rel_sep_div3
                FAC1_25PN = v_dot_v + 3.0 * GM_tot * rel_sep_div1
                FAC2_25PN = -(3.0 * v_dot_v + c17div3 * GM_tot * rel_sep_div1) * r_dot_v * rel_sep_div2

                ax += FAC0_25PN * (FAC1_25PN * dvx + FAC2_25PN * dx)
                ay += FAC0_25PN * (FAC1_25PN * dvy + FAC2_25PN * dy)
                az += FAC0_25PN * (FAC1_25PN * dvz + FAC2_25PN * dz)

                #printf("test FAC PN %g %g %g %g %g %g %g %g %g\n",FAC0_1PN,FAC1_1PN,FAC2_2PN,FAC0_2PN,FAC1_2PN,FAC2_2PN,FAC0_25PN,FAC1_25PN,FAC2_25PN);
                #print "values:", FAC0_1PN,FAC1_1PN,FAC2_2PN,FAC0_2PN,FAC1_2PN,FAC2_2PN,FAC0_25PN,FAC1_25PN,FAC2_25PN

            accx += converter.to_si(ax |nbody_system.length/nbody_system.time**2)
            accy += converter.to_si(ay |nbody_system.length/nbody_system.time**2)
            accz += converter.to_si(az |nbody_system.length/nbody_system.time**2)
        return accx, accy, accz

class BaseCode:
    def __init__(self, code, particles, eps=0|units.au):

        self._particles = particles
        m = self._particles.mass.sum()
        l = self._particles.position.length()
        self.converter = nbody_system.nbody_to_si(m, l)
        self.code = code(self.converter)
        self.code.parameters.epsilon_squared = eps**2

    def evolve_model(self, time):
        self.code.evolve_model(time)
    def copy_to_framework(self):
        self.channel_to_framework.copy()
    def get_gravity_at_point(self, r, x, y, z):
        return self.code.get_gravity_at_point(r, x, y, z)
    def get_potential_at_point(self, r, x, y, z):
        return self.code.get_potential_at_point(r, x, y, z)
    def get_timestep(self):
        return self.code.parameters.timestep
    @property
    def model_time(self):            
        return self.code.model_time
    @property
    def particles(self):
        return self.code.particles
    @property
    def total_energy(self):
        return self.code.kinetic_energy + self.code.potential_energy
    @property
    def stop(self):
        return self.code.stop

class Gravity(BaseCode):
    def __init__(self, code, particles, eps=0|units.au):
        BaseCode.__init__(self, code, particles, eps)
        self.code.particles.add_particles(self._particles)
        self.channel_to_framework \
            = self.code.particles.new_channel_to(self._particles)
        self.channel_from_framework \
            = self._particles.new_channel_to(self.code.particles)
        self.initial_total_energy = self.total_energy

def semi_to_orbital_period(a, Mtot) :
    return 2*math.pi * (a**3/(constants.G*Mtot)).sqrt()
    
def gravity_postNewton_bridge(Mprim, Msec, sma, ecc, t_end, n_steps):
                              
    stars = new_binary_from_orbital_elements(Mprim, Msec, sma, ecc,
                                             G=constants.G)
    converter = nbody_system.nbody_to_si(stars.mass.sum(), sma)
    stars.radius = 1|units.RSun
    stars.move_to_center()
    
    eps = 0.1 | units.AU
    gravity = Gravity(ph4, stars, eps)

    model_time = 0 | units.Myr
    filename = "gravPN.hdf5"
    write_set_to_file(stars.savepoint(model_time), filename, 'amuse',
                      overwrite_file=True,
                      append_to_file=False)

    postNewtonian_code = TidalFriction(gravity,
                                               (stars,),
                                               do_sync=True,
                                               verbose=False,
                                               radius_is_eps=False,
                                               h_smooth_is_eps=False,
                                               zero_smoothing=False)
    
    gravityPN = bridge.Bridge(use_threading=False)
    gravityPN.add_system(gravity,)
    gravityPN.add_code(postNewtonian_code)
    print("stars:", stars.mass.in_(units.MSun), sma.in_(units.RSun))
    P_orb  = semi_to_orbital_period(sma, stars.mass.sum())
    print("Porb=", P_orb.in_(units.day))
    gravityPN.timestep = 0.1 *P_orb

    a = [] | units.au
    e = []
    dt = t_end/n_steps
    while model_time < t_end:
        orbit = orbital_elements_from_binary(stars, G=constants.G)
        dE_gravity = gravity.initial_total_energy/gravity.total_energy
        print("Time:", model_time.in_(units.yr), \
              "ae=", orbit[2].in_(units.AU), orbit[3], \
              "dE=", dE_gravity)
        a.append(orbit[2])
        e.append(orbit[3])

        #print "dt=", gravityPN.timestep
        model_time += dt
        gravityPN.evolve_model(model_time)
        gravity.copy_to_framework()
        write_set_to_file(stars.savepoint(model_time), filename,
                          'amuse',
                          overwrite_file=True)
        #print "P=", model_time.in_(units.yr), gravity.particles.x.in_(units.au)
    gravity.stop()

    from mathplotlib import pyplot
    pyplot.scatter(a, e)
    pyplot.show()

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-n", dest="n_steps", type="int", default = 100,
                      help="number of diagnostics time steps [%default]")
    result.add_option("--Mprim", unit=units.MSun,
                      dest="Mprim", type="float", default = 1.441|units.MSun,
                      help="Mass of the primary star [%default]")
    result.add_option("--Msec", unit=units.MSun,
                      dest="Msec", type="float", default = 1.441|units.MSun,
                      help="Mass of the secondary star [%default]")
    result.add_option("-a", unit=units.AU,
                      dest="sma", type="float", default =  1950100|units.km,
                      help="initial orbital separation [%default]")
    result.add_option("-e", dest="ecc", type="float", default =  0.6171334,
                      help="initial orbital eccentricity [%default]")
    result.add_option("-t", unit=units.yr, 
                      dest="t_end", type="float", default = 10.0|units.yr,
                      help="end time of the simulation [%default]")
    return result

if __name__ in ('__main__', '__plot__'):
    o, arguments  = new_option_parser().parse_args()
    numpy.random.seed(123)
    gravity_postNewton_bridge(**o.__dict__)
