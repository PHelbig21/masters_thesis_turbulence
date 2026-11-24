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
import helpful_functions as hf # helpful functions for spectral grid handling

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

    def __init__(self, i_seed, f_seed): # TO BE FINISHED


        # ---- parameters ---- DONE

        # hardcoded
        self.R = 1 # no need to change, this is hardcoded

        # timestep
        if par.path_wise_forcing: # if pathwise forcing is activated
            self.timestep = par.a * par.timestep_min
        else: # usually (without pathwise forcing)
            self.timestep = par.timestep


        # ---- basis ---- DONE

        self.coords = d3.S2Coordinates('phi', 'theta')
        self.dist = d3.Distributor(self.coords, dtype=par.dtype)
        self.basis = d3.SphereBasis(self.coords, (par.Nphi, par.Ntheta), radius=self.R, dealias=par.dealias, dtype=par.dtype) 

        # ---- spectral grids ---- DONE

        # find global shape
        shape = self.basis.global_shape(grid_space=(False, False), scales=(1, 1))
        # find global dedalus spectral grid indices
        i_s = np.arange(shape[0])
        j_s = np.arange(shape[1])
        # create global spectral ij-grid
        i_grid, j_grid = np.meshgrid(i_s, j_s, indexing='ij')
        # find slicing to extract local spectral grid
        field = self.dist.Field(name='field', bases=self.basis)
        local_slices = field.domain.dist.coeff_layout.slices(field.domain, scales=1)
        local_slice = local_slices[0]
        i_grid = i_grid[local_slice]
        j_grid = j_grid[local_slice]

        # spectral grid (grid indexing internally used by Dedalus)
        self.m_grid, self.ell_grid = self.basis.elements_to_groups((False, False), (i_grid, j_grid))
        # those grids save the l and m values on the spectral grid ordering provided by Dedalus

        # thinned mode grid (without spins)
        self.ell_grid_thinned = self.ell_grid[::2]
        self.m_grid_thinned = self.m_grid[::2]



        # ---- initialise rngs ---- DONE

        self.i_seed = i_seed # seed for initial conditions
        self.f_seed = f_seed # seed for forcing

        self.i_rng = self._set_initial_rng() # rng for initial conditions
        self.f_rng = self._set_forcing_rng() # rng for forcing


        # ---- calculate standard deviation fields ---- DONE

        # computes the field of standard deviations of the stream function coefficients from the given energy spectrum
        self._compute_i_sigma()
        # computes the field of standard deviations of the forcing stream function coefficients from the given forcing spectrum
        self._compute_f_sigma()


        # ---- initialise external fields ---- DONE

        # velocity
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.basis)
        # pressure
        self.p = self.dist.Field(name='p', bases=self.basis)
        # tau field (not changes within this class, just initialised here)
        self.tau_p = self.dist.Field(name='tau_p')
        # forcing
        self.f = self.dist.VectorField(self.coords, name='f', bases=self.basis)


        # ---- initialise internal fields ---- DONE

        # reserves storage for the velocity stream function
        self._s = self.dist.Field(name='s', bases=self.basis)
        # reserves storage for the forcing stream function
        self._sf = self.dist.Field(name='sf', bases=self.basis)


        # ---- initialise the pressure solver ---- DONE

        # build helper fields
        # self._nonlinear = self.dist.VectorField(self.coords, bases=(self.xbasis, self.ybasis))
        self._rhs_p = self.dist.Field(name='rhs_p', bases=self.basis)
        self._tau_lap = self.dist.Field(name='tau_lap')

        # build poisson solver
        poisson_problem = d3.LBVP([self.p, self._tau_lap], namespace={'p': self.p, 'tau_lap': self._tau_lap, 'rhs_p': self._rhs_p})
        poisson_problem.add_equation("lap(p) + tau_lap = rhs_p")
        poisson_problem.add_equation("ave(p) = 0")
        self._poisson_solver = poisson_problem.build_solver()




    # _____ INTERNAL METHODS _____
    # internal methods are called within the class


    # ---- create random number generators ----


    def _set_initial_rng(self): # DONE

        '''
        creates a new random number generator for initial conditions
        '''

        # create random number generator for initial conditions
        # add rank of processor to prevent each core using the same values
        return(np.random.default_rng(self.i_seed + comm.Get_rank()))


    def _set_forcing_rng(self): # DONE (but not for pathwise forcing)

        '''
        creates a new random number generator for forcing
        '''

        # if we want pathwise forcing, we must make forcing robust against parallelisation
        if par.path_wise_forcing: # TODO, has not yet been implemented for the sphere

            print("WARNING: pathwise forcing not yet implemented for the sphere")
            
            # placeholder

            return(0) # this is a field of rngs

        # if we don't want pathwise forcing, we don't care about parallelisation
        else:
            # use the seed to create a random number generator for usual forcing
            # add rank of processor to prevent each core using the same values
            return(np.random.default_rng(self.f_seed + comm.Get_rank())) # this is a single rng




    # ---- compute standard deviation fields ----


    def _compute_i_sigma(self): # in __init__ aufgerufen # DONE

        """
        computes the standard deviation of the stream function coefficients
        according to a given energy spectrum (here we use the one given in Herring 1974)
        gives a global lm field
        """

        # simplifications (for better readability)
        ell = self.ell_grid_thinned
        ell0 = par.ell0
        u0 = par.u0
        R = self.R

        # initialise sigma (filled with zeros)
        Eq = np.zeros_like(ell, dtype=par.dtype) # create empty array for the average energy spectrum
        sigma = np.zeros_like(ell, dtype=par.dtype) # create empty array for sigma

        if par.i_kind == 'herring':
            # fill sigma field according to the initial energy spectrum (Herring 1974)
            Eq[ell != 0] = u0**2*ell[ell != 0] / (2*ell0**2) * np.exp(-ell[ell != 0]/ell0)

        elif par.i_kind == 'segalini':
            # fill sigma field according to the initial energy spectrum (Segalini & Lindborg)
            Eq[ell != 0] = u0**2 / (2*par.sigmai*np.sqrt(2*np.pi)) * np.exp(-((ell[ell != 0]-ell0)/par.sigmai)**2/2)


        # fill sigma field according to the initial energy spectrum
        sigma[ell != 0] = R * np.sqrt(Eq[ell != 0] / (ell[ell != 0]**3 + 3/2*ell[ell != 0]**2 + 1/2*ell[ell != 0]))
        sigma[ell == 0] = 0


        
        # set initial sigma 
        self.i_sigma = sigma


    def _compute_f_sigma(self): # in __init__ aufgerufen # DONE

        """
        computes the standard deviation of the forcing stream function coefficients
        according to a given forcing spectrum (Gaussian distributed)
        """

        # simplifications (for better readability)
        ell = self.ell_grid_thinned
        R = self.R
        ellf = par.ellf*np.ones_like(ell) # transform scalar forcing number to a grid (needed for subtraction from the q grid below)
        sigmaf = par.sigmaf
        pf = par.pf

        # initialise sigma
        Eq = np.zeros_like(ell, dtype=par.dtype)
        sigma = np.zeros_like(ell, dtype=par.dtype) # create empty array for sigma

        # forcing spectrum
        Eq[ell != 0] = pf*ell[ell != 0] / (np.sqrt(2*np.pi)*sigmaf*ellf[ell != 0]) * np.exp(-(((ell[ell != 0]-ellf[ell != 0])/sigmaf)**2)/2)

        # fill sigma field according to the gaussian distributes forcing spectrum
        sigma[ell != 0] = R * np.sqrt(Eq[ell != 0] / (ell[ell != 0]**3 + 3/2*ell[ell != 0]**2 + 1/2*ell[ell != 0]))
        sigma[ell == 0] = 0 # zero mean

        # set forcing sigma 
        self.f_sigma = sigma



    # ---- draw stream function coefficients ----


    def _stable_seed(self, base_seed, i, j): # TODO but only needed for pathwise forcing, so leave it for now
        
        # placeholder

        return 0


    def _create_stream_coefficients(self, rng, sigma): # DONE # called by create_initial_velocity and create_forcing

        """
        Draws random stream coefficients according to the given standard deviation
        and builds the full stream function coefficient field.

        Note that we draw the amplitudes and phases, these must be translated into the notation that Dedalus uses.
        For more informaton see Arbeitsgruppenpraktikum.
        """

        # simplification
        ell = self.ell_grid_thinned
        m = self.m_grid_thinned

        # draw coefficients from sigma field (Psi and Phi are thinned fields)
        Psi = rng.normal(loc=0, scale=sigma) # $\Psi_{nm}^0$ amplitude of complex wave coefficient (m>=0), see Arbeitsgruppenpraktikum Eq. (36)
        Phi = rng.uniform(0, 2*np.pi, size=ell.shape) # $\Phi_{nm}$ phase of complex wave coefficient (m>=0), see Arbeitsgruppenpraktikum Eq. (36)

        # calculate stream function coefficients from amplitudes and phases (sp and sm are thinned fields)
        # see Arbeitsgruppenpraktikum Eq. (37) and Eq. (38)
        # spin +1
        sp = np.zeros_like(ell, dtype=par.dtype)
        sp[m == 0] = 1/np.sqrt(2*np.pi)*Psi[m == 0] * np.sqrt(4*np.pi)
        sp[m != 0] = np.power(-1,m[m != 0])*2/(np.sqrt(2*np.pi))*Psi[m != 0]*np.cos(Phi[m != 0]) * np.sqrt(4*np.pi)
        # spin -1
        sm = np.zeros_like(ell, dtype=par.dtype)
        sm[m == 0] = 0
        sm[m != 0] = -np.power(-1,m[m != 0])*2/(np.sqrt(2*np.pi))*Psi[m != 0]*np.sin(Phi[m != 0]) * np.sqrt(4*np.pi)

        # put sp and sm together into a single stream function coefficient field
        s_c = np.zeros_like(self.ell_grid, dtype=par.dtype)
        s_c[0::2] = sp
        s_c[1::2] = sm
        
        # return final coefficient field of stream function (has already the shape of a Dedalus field, so s['c'] = s_c is possible)
        return s_c



    def _create_stream_coefficients_pathwise(self, rngs, sigma): # TODO, but only needed for pathwise forcing, so leave it for now # called by create_initial_velocity and create_forcing

        # placeholder

        return 0



    # ---- pressure solver ----


    def _find_pressure(self): # DONE

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


    def create_initial_velocity(self): # DONE, no changes

        """
        creates a velocity field u and its according pressure field
        with a random initial condition using the random number generator self.i_rng
        """

        # fill stream function with random coefficients
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


    def create_forcing(self): # DONE. no changes, but pathwise forcing does not work for now

        """
        creates a random forcing field f
        using the random number generator self.f_rng
        """

        # fill stream function with random coefficients
        if par.path_wise_forcing:
            # this implementation is independent of parallelisation and gains the stream function from multiple intermediate forcing steps
            # this implementation is slow
            self._sf['c'] = self._create_stream_coefficients_pathwise(self.f_rng, self.f_sigma) # here, self.f_rng is a field of rngs
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