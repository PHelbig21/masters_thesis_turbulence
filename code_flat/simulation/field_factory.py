'''
###################################################################################################################

FIELD FACTORY

This file contains the class FieldFactory.
It contains the methods for the creation of the initial conditions and the forcing.

Overview:
1. Imports
2. Field Factory
    - Initialisation
    - Internal Methods
    - External Methods

###################################################################################################################
'''



# __________ IMPORTS __________

# files
import parameters as par # parameter file

# calculation
import numpy as np
import dedalus.public as d3  # to create dedalus fields

# parallelisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
import hashlib # for _stable_seed


# __________ FIELD FACTORY __________

class FieldFactory():

    """
    Encapsulates spectral grid data and provides methods for
    initial conditions and forcing.
    """

    # _____ INITIALISATION _____

    def __init__(self, i_seed, f_seed):


        # ---- parameters ----

        # hardcoded
        self.L = 2*np.pi # no need to change, this is hardcoded

        # timestep
        if par.path_wise_forcing: # if pathwise forcing is activated
            self.timestep = par.a * par.timestep_min
        else: # usually (without pathwise forcing)
            self.timestep = par.timestep


        # ---- basis ----

        self.coords = d3.CartesianCoordinates('x', 'y')
        self.dist = d3.Distributor(self.coords, dtype=par.dtype)
        self.xbasis = d3.RealFourier(self.coords['x'], size=par.N, bounds=(0, self.L), dealias=par.dealias)
        self.ybasis = d3.RealFourier(self.coords['y'], size=par.N, bounds=(0, self.L), dealias=par.dealias)


        # ---- spectral grids ----

        # spectral grid (grid indexing internally used by Dedalus)
        kx = self.dist.local_modes(self.xbasis)
        ky = self.dist.local_modes(self.ybasis)
        # kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij') # spectral grid

        # wave number grid (grid indexing according to the physical mode number)
        self.qx = (kx//2)
        self.qy = (ky//2)
        qx_grid, qy_grid = np.meshgrid(self.qx, self.qy, indexing='ij') # fields of wave vector components
        self.q_grid = np.sqrt(qx_grid**2 + qy_grid**2) # field of absolute of wave vector

        # thinned wave number grid
        self.q_grid_thinned = self.q_grid[::2, ::2] # grid for only one coefficient field (instead of four)
        self.qx_grid_thinned = qx_grid[::2, ::2]
        self.qy_grid_thinned = qy_grid[::2, ::2] # self. davor, wenn es später in den Methoden gebraucht wird

        # truncation wave number
        local_max_qx = np.max(self.qx) # find maximum of qx on each core
        local_max_qy = np.max(self.qy) # find maximum of qy on each core
        self.trunc = min(comm.allreduce(local_max_qx, op=MPI.MAX), comm.allreduce(local_max_qy, op=MPI.MAX)) # find maximum qx and qy on all cores, the truncation wave mode is then the minimum of the two


        # ---- initialise rngs ----

        self.i_seed = i_seed # seed for initial conditions
        self.f_seed = f_seed # seed for forcing

        self.i_rng = self._set_initial_rng() # rng for initial conditions
        self.f_rng = self._set_forcing_rng() # rng for forcing


        # ---- calculate standard deviation fields ----

        # computes the field of standard deviations of the stream function coefficients from the given energy spectrum
        self._compute_i_sigma()
        # computes the field of standard deviations of the forcing stream function coefficients from the given forcing spectrum
        self._compute_f_sigma()


        # ---- initialise external fields ----

        # velocity
        self.u = self.dist.VectorField(self.coords, name='u', bases=(self.xbasis, self.ybasis))
        # pressure
        self.p = self.dist.Field(name='p', bases=(self.xbasis, self.ybasis))
        # tau field (not changes within this class, just initialised here)
        self.tau_p = self.dist.Field(name='tau_p')
        # forcing
        self.f = self.dist.VectorField(self.coords, name='f', bases=(self.xbasis, self.ybasis))


        # ---- initialise internal fields ----

        # reserves storage for the velocity stream function
        self._s = self.dist.Field(name='s', bases=(self.xbasis,self.ybasis))
        # reserves storage for the forcing stream function
        self._sf = self.dist.Field(name='sf', bases=(self.xbasis,self.ybasis))


        # ---- initialise the pressure solver ----

        # build helper fields
        # self._nonlinear = self.dist.VectorField(self.coords, bases=(self.xbasis, self.ybasis))
        self._rhs_p = self.dist.Field(name='rhs_p', bases=(self.xbasis, self.ybasis))
        self._tau_lap = self.dist.Field(name='tau_lap')

        # build poisson solver
        poisson_problem = d3.LBVP([self.p, self._tau_lap], namespace={'p': self.p, 'tau_lap': self._tau_lap, 'rhs_p': self._rhs_p})
        poisson_problem.add_equation("lap(p) + tau_lap = rhs_p")
        poisson_problem.add_equation("integ(p) = 0")
        self._poisson_solver = poisson_problem.build_solver()



    # _____ INTERNAL METHODS _____
    # internal methods are called within the class


    # ---- create random number generators ----


    def _set_initial_rng(self):

        '''
        creates a new random number generator for initial conditions
        '''

        # if we want to do spatial convergence tests, we must ensure that the initial conditions are reproducible, independent of the resolution
        # so we create a field of random number generators, such that each modes has its own reproducable random number generator
        if par.resolution_independence: #or par.path_wise_forcing:

            # build grid of rngs, one rng for each point in maximal spectral space
            rngs = np.empty_like(self.q_grid_thinned, dtype=object)

            # fill the field of rngs with rngs, using the given seed
            for i in range(rngs.shape[0]): # loop over all qx
                for j in range(rngs.shape[1]): # loop over all qy
                    qx = self.qx_grid_thinned[i, j] # get x-coordinate of mode
                    qy = self.qy_grid_thinned[i, j] # get y-coordinate of mode
                    this_seed = self._stable_seed(self.i_seed, qx, qy) # from seed create an individual seed for (qx, qy) (this seed is reproducable and always gives the same rng for the same i_seed, qx and qy)
                    rngs[i, j] = np.random.default_rng(this_seed) # create and individual rng for (qx, qy)

            return(rngs) # this is a field of rngs that has the shape of self.q_grid_thinned, but will always give the same rng for the same mode, independent of the parallelisation and of how large the spectral space is.

        else:

            # create random number generator for initial conditions
            # add rank of processor to prevent each core using the same values
            return(np.random.default_rng(self.i_seed + comm.Get_rank()))


    def _set_forcing_rng(self):

        '''
        creates a new random number generator for forcing
        '''

        # if we want pathwise forcing, we must make forcing robust against parallelisation
        # also if we want to do spatial convergence tests, we must ensure that the forcing is independent of the spatial resolution, so each mode has its own reproducable random number generator
        if par.path_wise_forcing or par.resolution_independence:
            
            # create a field of rngs, one rng for each point in spectral space
            rngs = np.full_like(self.q_grid_thinned, fill_value=None, dtype=object)

            # fill the field of rngs with rngs, using the given seed
            for i in range(rngs.shape[0]): # loop over all qx
                for j in range(rngs.shape[1]): # loop over all qy
                    qx = self.qx_grid_thinned[i, j] # get x-coordinate of mode
                    qy = self.qy_grid_thinned[i, j] # get y-coordinate of mode
                    this_seed = self._stable_seed(self.f_seed, qx, qy) # from seed create an individual seed for (qx, qy)
                    rngs[i,j] = np.random.default_rng(this_seed) # create and individual rng for (qx, qy)

            return(rngs) # this is a field of rngs

        # if we don't want pathwise forcing, we don't care about parallelisation
        else:
            # use the seed to create a random number generator for usual forcing
            # add rank of processor to prevent each core using the same values
            return(np.random.default_rng(self.f_seed + comm.Get_rank())) # this is a single rng



    # ---- compute standard deviation fields ----


    def _compute_i_sigma(self): # in __init__ aufgerufen

        """
        computes the standard deviation of the stream function coefficients
        according to a given energy spectrum (here we use the one given in Herring 1974)
        """

        # simplifications (for better readability)
        q = self.q_grid_thinned
        q0 = par.q0
        u0 = par.u0
        L = self.L

        # initialise sigma
        Eq = np.zeros_like(q, dtype=par.dtype) # create empty array for the average energy spectrum
        sigma = np.zeros_like(q, dtype=par.dtype) # create empty array for sigma

        if par.i_kind == 'herring':
            # fill sigma field according to the initial energy spectrum (Herring 1974)
            Eq[q != 0] = u0**2*q[q != 0]/(2*q0**2) * np.exp(-q[q != 0]/q0)
        
        elif par.i_kind == 'segalini':
            # fill sigma field according to the initial energy spectrum (Segalini & Lindborg)
            Eq[q != 0] = u0**2/(2*par.sigmai*np.sqrt(2*np.pi)) * np.exp(-((q[q != 0]-q0)/par.sigmai)**2/2)

        else:
            print("Please choose a valid initial condition")

        sigma[q != 0] = L/(2*np.pi) * np.sqrt(Eq[q != 0]/(np.pi*q[q != 0]**3))
        sigma[q == 0] = 0 # zero mean
        
        # spherical truncation of sigma for isotropy
        sigma[q > self.trunc] = 0 # spherical truncation, truncate wave modes above truncation wave mode

        # set initial sigma 
        self.i_sigma = sigma


    def _compute_f_sigma(self): # in __init__ aufgerufen

        """
        computes the standard deviation of the forcing stream function coefficients
        according to a given forcing spectrum (Gaussian distributed)
        """

        # simplifications (for better readability)
        q = self.q_grid_thinned
        L = self.L 
        qf = par.qf*np.ones_like(q) # transform scalar forcing number to a grid (needed for subtraction from the q grid below)
        sigmaf = par.sigmaf
        pf = par.pf

        # initialise sigma
        Eq = np.zeros_like(q, dtype=par.dtype)
        sigma = np.zeros_like(q, dtype=par.dtype) # create empty array for sigma

        # forcing spectrum
        Eq[q != 0] = pf*q[q != 0] / (np.sqrt(2*np.pi)*sigmaf*qf[q != 0]) * np.exp(-(((q[q != 0]-qf[q != 0])/sigmaf)**2)/2)
 
        # fill sigma field according to the gaussian distributes forcing spectrum
        sigma[q != 0] = L/(2*np.pi) * np.sqrt(Eq[q != 0]/(np.pi*q[q != 0]**3))
        sigma[q == 0] = 0 # zero mean

        # spherical truncation of sigma for isotropy
        sigma[q > self.trunc] = 0 # spherical truncation, truncate wave modes above truncation wave mode

        # set forcing sigma 
        self.f_sigma = sigma



    # ---- draw stream function coefficients ----


    def _stable_seed(self, base_seed, i, j):
        
        '''
        creates a random number from the base_seed and the indices i and j
        this number is used in set_forcing_seed to create a field of rngs used in create_forcing
        to make the forcing independent of the parallelisation
        '''
        
        s = f"{base_seed}_{i}_{j}".encode()

        return int(hashlib.sha256(s).hexdigest()[:8], 16)


    def _create_stream_coefficients(self, rng, sigma): # called by create_initial_velocity and create_forcing

        """
        Draws random stream coefficients according to the given standard deviation
        and builds the full stream function coefficient field.

        Note that we draw the amplitudes and phases, these must be translated into the notation that Dedalus uses.
        For more informaton see Arbeitsgruppenpraktikum.
        """

        # simplification
        q = self.q_grid_thinned
        qx = self.qx_grid_thinned
        qy = self.qy_grid_thinned

        # fields of amplitudes (see Arbeitsgruppenpraktikum)
        s0_pp = rng.normal(loc=0, scale=sigma) # create normal distributed fields of amplitudes with given standard deviation sigma using a costum random number generator rng
        s0_pm = rng.normal(loc=0, scale=sigma) # create normal distributed fields of amplitudes with given standard deviation sigma using a costum random number generator rng

        # fields of phases (see Arbeitsgruppenpraktikum)
        phase_pp = rng.uniform(0, 2*np.pi, size=q.shape) # create uniform distributed fields of phases
        phase_pm = rng.uniform(0, 2*np.pi, size=q.shape) # create uniform distributed fields of phases

        # transform spectral space coefficients (see Arbeitsgruppenpraktikum)
        # for qx>0 and qy>0
        s_pp = 2*s0_pp*np.cos(phase_pp) + 2*s0_pm*np.cos(phase_pm)
        s_pm = 2*s0_pp*np.sin(phase_pp) - 2*s0_pm*np.sin(phase_pm)
        s_mp = 2*s0_pp*np.sin(phase_pp) + 2*s0_pm*np.sin(phase_pm)
        s_mm = -2*s0_pp*np.cos(phase_pp) + 2*s0_pm*np.cos(phase_pm)
        # for qx=qy=0
        s_pp[q==0] = s0_pp[q==0]
        s_pm[q==0] = 0.
        s_mp[q==0] = 0.
        s_mm[q==0] = 0.
        # for qx=0, qy>0
        s_pp[qx==0] = 2*s0_pp[qx==0]*np.cos(phase_pp[qx==0])
        s_pm[qx==0] = 2*s0_pp[qx==0]*np.sin(phase_pp[qx==0]) # 0.
        s_mp[qx==0] = 0 #2*s0_pp[qx==0]*np.sin(phase_pp[qx==0])
        s_mm[qx==0] = 0.
        # for qx>0, qy=0
        s_pp[qy==0] = 2*s0_pp[qy==0]*np.cos(phase_pp[qy==0])
        s_pm[qy==0] = 0. # 2*s0_pp[qy==0]*np.sin(phase_pp[qy==0])
        s_mp[qy==0] = 2*s0_pp[qy==0]*np.sin(phase_pp[qy==0]) # 0.
        s_mm[qy==0] = 0.

        # rearrange coefficients in grid to Dedalus notation
        s_c = np.zeros_like(self.q_grid)
        s_c[0::2, 0::2] = s_pp # same as s_c[2*i, 2*j] = s_pp[i, j] etc. but faster without loop
        s_c[0::2, 1::2] = s_pm # same as s_c[2*i,   2*j+1] = s_pm[i, j] etc.
        s_c[1::2, 0::2] = s_mp
        s_c[1::2, 1::2] = s_mm

        # return final coefficient field of stream function (has already the shape of a Dedalus field, so s['c'] = s_c is possible)
        return s_c


    def _create_stream_coefficients_pathwise(self, rngs, sigma): # called by create_initial_velocity and create_forcing

        """
        creates stream function coefficients, which follow a fixed Wiener process,
        thus this Wiener process is reproducable (if the same seed is used) over simulations with different timesteps
        this function is MPI robust
        """

        # simplification
        q = self.q_grid_thinned
        qx = self.qx_grid_thinned
        qy = self.qy_grid_thinned

        # create total stream function (then contributions s_c from each intermediate forcing time k can be added)
        s_c_total = np.zeros_like(self.q_grid)    

        # create empty fields of amplitudes and phases
        s0_pp = np.zeros_like(sigma)
        s0_pm = np.zeros_like(sigma)
        phase_pp = np.zeros_like(sigma)
        phase_pm = np.zeros_like(sigma)

        # loop over all intermediate forcing times within the timestep
        for k in range(par.a):         

            # fill fields of amplitudes and phases
            # this part is MPI robust (assignment of same random values to same locations, independent of parallelisation)
            for i in range(sigma.shape[0]):
                for j in range(sigma.shape[1]):
                    # fields of amplitudes
                    s0_pp[i, j] = rngs[i, j].normal(loc=0, scale=sigma[i,j])
                    s0_pm[i, j] = rngs[i, j].normal(loc=0, scale=sigma[i,j])
                    # fields of phases
                    phase_pp[i, j] = rngs[i, j].uniform(0, 2*np.pi)
                    phase_pm[i, j] = rngs[i, j].uniform(0, 2*np.pi)

            # transform spectral space coefficients
            # for qx>0 and qy>0
            s_pp = 2*s0_pp*np.cos(phase_pp) + 2*s0_pm*np.cos(phase_pm)
            s_pm = 2*s0_pp*np.sin(phase_pp) - 2*s0_pm*np.sin(phase_pm)
            s_mp = 2*s0_pp*np.sin(phase_pp) + 2*s0_pm*np.sin(phase_pm)
            s_mm = -2*s0_pp*np.cos(phase_pp) + 2*s0_pm*np.cos(phase_pm)
            # for qx=qy=0
            s_pp[q==0] = s0_pp[q==0]
            s_pm[q==0] = 0.
            s_mp[q==0] = 0.
            s_mm[q==0] = 0.
            # for qx=0, qy>0
            s_pp[qx==0] = 2*s0_pp[qx==0]*np.cos(phase_pp[qx==0])
            s_pm[qx==0] = 2*s0_pp[qx==0]*np.sin(phase_pp[qx==0]) # 0.
            s_mp[qx==0] = 0. # 2*s0_pp[qx==0]*np.sin(phase_pp[qx==0])
            s_mm[qx==0] = 0.
            # for qx>0, qy=0
            s_pp[qy==0] = 2*s0_pp[qy==0]*np.cos(phase_pp[qy==0])
            s_pm[qy==0] = 0. # 2*s0_pp[qy==0]*np.sin(phase_pp[qy==0])
            s_mp[qy==0] = 2*s0_pp[qy==0]*np.sin(phase_pp[qy==0]) # 0.
            s_mm[qy==0] = 0.

            # rearrange coefficients in grid
            s_c = np.zeros_like(self.q_grid)
            s_c[0::2, 0::2] = s_pp
            s_c[0::2, 1::2] = s_pm
            s_c[1::2, 0::2] = s_mp
            s_c[1::2, 1::2] = s_mm

            # add current stream contribution to total stream function
            s_c_total += s_c/np.sqrt(par.a) # (/np.sqrt(par.a) because of averaging, see Methodik)

        # return final coefficient field of stream function (has already the shape of a Dedalus field, so s['c'] = s_c_total is possible)
        return s_c_total


    def _create_stream_coefficients_resindependent(self, rngs, sigma):

        """
        Draws random stream coefficients according to the given standard deviation
        and builds the full stream function coefficient field.
        Since in this function each mode has its own reproducable rng (independent of resolution and parallelisation),
        these stream function coefficients are independent of the spatial resolution, so they can be used for spatial convergence tests.
        """

        # simplification
        q = self.q_grid_thinned
        qx = self.qx_grid_thinned
        qy = self.qy_grid_thinned

        # create empty fields of amplitudes and phases
        s0_pp = np.zeros_like(sigma)
        s0_pm = np.zeros_like(sigma)
        phase_pp = np.zeros_like(sigma)
        phase_pm = np.zeros_like(sigma)

        # fill fields of amplitudes and phases
        # this part is MPI robust (assignment of same random values to same locations, independent of parallelisation)
        for i in range(sigma.shape[0]):
            for j in range(sigma.shape[1]):
                # fields of amplitudes
                s0_pp[i, j] = rngs[i, j].normal(loc=0, scale=sigma[i,j])
                s0_pm[i, j] = rngs[i, j].normal(loc=0, scale=sigma[i,j])
                # fields of phases
                phase_pp[i, j] = rngs[i, j].uniform(0, 2*np.pi)
                phase_pm[i, j] = rngs[i, j].uniform(0, 2*np.pi)

        # transform spectral space coefficients (see Arbeitsgruppenpraktikum)
        # for qx>0 and qy>0
        s_pp = 2*s0_pp*np.cos(phase_pp) + 2*s0_pm*np.cos(phase_pm)
        s_pm = 2*s0_pp*np.sin(phase_pp) - 2*s0_pm*np.sin(phase_pm)
        s_mp = 2*s0_pp*np.sin(phase_pp) + 2*s0_pm*np.sin(phase_pm)
        s_mm = -2*s0_pp*np.cos(phase_pp) + 2*s0_pm*np.cos(phase_pm)
        # for qx=qy=0
        s_pp[q==0] = s0_pp[q==0]
        s_pm[q==0] = 0.
        s_mp[q==0] = 0.
        s_mm[q==0] = 0.
        # for qx=0, qy>0
        s_pp[qx==0] = 2*s0_pp[qx==0]*np.cos(phase_pp[qx==0])
        s_pm[qx==0] = 2*s0_pp[qx==0]*np.sin(phase_pp[qx==0]) # 0.
        s_mp[qx==0] = 0. # 2*s0_pp[qx==0]*np.sin(phase_pp[qx==0])
        s_mm[qx==0] = 0.
        # for qx>0, qy=0
        s_pp[qy==0] = 2*s0_pp[qy==0]*np.cos(phase_pp[qy==0])
        s_pm[qy==0] = 0. # 2*s0_pp[qy==0]*np.sin(phase_pp[qy==0])
        s_mp[qy==0] = 2*s0_pp[qy==0]*np.sin(phase_pp[qy==0]) # 0.
        s_mm[qy==0] = 0.

        # rearrange coefficients in grid to Dedalus notation
        s_c = np.zeros_like(self.q_grid)
        s_c[0::2, 0::2] = s_pp # same as s_c[2*i, 2*j] = s_pp[i, j] etc. but faster without loop
        s_c[0::2, 1::2] = s_pm # same as s_c[2*i,   2*j+1] = s_pm[i, j] etc.
        s_c[1::2, 0::2] = s_mp
        s_c[1::2, 1::2] = s_mm

        # return final coefficient field of stream function (has already the shape of a Dedalus field, so s['c'] = s_c is possible)
        return s_c




    # ---- pressure solver ----


    def _find_pressure(self):

        '''
        Finds the pressure field p from a given velocity field u by solving the Poisson equation.
        '''

        # Evaluate right-hand side of Poisson equation: ∇²p = -(∇(u · ∇u))

        rhs_p = (-d3.div(self.u @ d3.grad(self.u))).evaluate() # -∇(u · ∇u)
        rhs_p.change_scales(1)
        self._rhs_p['g'] = rhs_p['g'] 

        
        # Solve Poisson equation to get initial pressure field:
        self._poisson_solver.solve()




    # _____ EXTERNAL METHODS _____
    # external methods are also called from outside (directly in simulation.py)


    # ----- initial conditions ----


    def create_initial_velocity(self):

        """
        creates a velocity field u and its according pressure field
        with a random initial condition using the random number generator self.i_rng
        """

        # fill stream function with random coefficients

        if par.resolution_independence: #or par.path_wise_forcing:
            # this implementation is independent of parallelisation and spatial resolution
            self._s['c'] = self._create_stream_coefficients_resindependent(self.i_rng, self.i_sigma) # here, self.f_rng is a field of rngs
        else:
            # this implementation give different initial conditions for different resolutions despite the same i_seed
            self._s['c'] = self._create_stream_coefficients(self.i_rng, self.i_sigma)

        # transform stream function to velocity
        helper = (-d3.skew(d3.Gradient(self._s, self.coords))).evaluate()
        helper.change_scales(1)
        self.u['g'] = helper['g'] 

        # find respective pressure field
        self._find_pressure() # this function modifies p such that it corresponds to the velocity field u

        # return initial velocity field
        return(self.u, self.p, self.tau_p)



    # ----- forcing ----


    def create_forcing(self):

        """
        creates a random forcing field f
        using the random number generator self.f_rng
        """

        # fill stream function with random coefficients
        if par.path_wise_forcing:
            # this implementation is independent of parallelisation and gains the stream function from multiple intermediate forcing steps
            # this implementation is slow
            self._sf['c'] = self._create_stream_coefficients_pathwise(self.f_rng, self.f_sigma) # here, self.f_rng is a field of rngs
        elif par.resolution_independence:
            # this implementation is independent of parallelisation and spatial resolution
            # this implementation is slow
            self._sf['c'] = self._create_stream_coefficients_resindependent(self.f_rng, self.f_sigma) # here, self.f_rng is a field of rngs
        else:
            # this implementation depends on parallelisation and gains the stream function from a single forcing step
            # this implementation is fast
            self._sf['c'] = self._create_stream_coefficients(self.f_rng, self.f_sigma) # here, self.f_rng is a single rng
        
        # transform stream function to forcing
        helper = (-d3.skew(d3.Gradient(self._sf, self.coords))).evaluate()
        helper.change_scales(1) # otherwise shape conflict in next line
        self.f['g'] = helper['g']/(np.sqrt(self.timestep)) # rescaling of the forcing with the timestep to replace discontinous forcing with constant value such that the Wiener increment is the same, see Methodik

        # return forcing field
        return(self.f)



    
