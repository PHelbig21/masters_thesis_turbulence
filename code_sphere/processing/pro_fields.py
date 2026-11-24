
'''
###################################################################################################################

PROCESSING FIELDS

This file contains the class ProcessingFields.
It contains the methods for reading, processing the raw data and saving the processed data.

Overview:
1. Imports
2. Processing fields
    - initialisation
    - internal methods
        - _theoretical_quantities
    - external methods
        - calculations
        - saving

###################################################################################################################
'''



# __________ IMPORTS __________


# external files
import pro_parameters as pp
import pro_functions as pf

# file management
import os
import h5py

# calculation
import numpy as np
import dedalus.public as d3  # to create dedalus fields
from scipy.special import jv # bessel function
from scipy.signal import savgol_filter # filter to remove fluctuations in Pi_V, Pi_E and V
from scipy.special import hyp2f1
import findiff # finite differences to calculate derivatives
import time # time measurement

# parallelisation for Sp
from joblib import Parallel, delayed



# __________ PROCESSING FIELDS __________


class ProcessingFields():

    '''
    class that saves all field data
    - raw data (vorticity, pressure)
    - grid fields
    - processed fields ()
    '''


    # _____ INITIALISATION _____


    def __init__(self):

        # ---- access hdf5 file ---- DONE

        self.processed_path = os.path.join(pp.run_dir, f"run_{pp.run_id:05d}", "processed_data", f"run_{pp.run_id:05d}.h5") # create path to .h5 file
    

        # ---- access simulation parameters ---- DONE
        
        # (stored in hdf5-file for processed data)
        self.par_path = os.path.join(pp.run_dir, "run_{:05d}/processed_data/run_{:05d}.h5".format(pp.run_id, pp.run_id))
        
        # read and store simulation parameters
        with h5py.File(self.par_path, mode='r') as file:

            # COMPUTATIONAL PARAMETERS
            dtype_str = file["parameters/dtype"][()].decode() # dtype_str = "<class 'numpy.float64'>"
            self.dtype = np.dtype(dtype_str.split("'")[1].split(".")[-1])
            self.dealias = file["parameters/dealias"][()]

            # PHYSICAL PARAMETERS
            self.R = 1 # hardcoded
            self.Nphi = file["parameters/Nphi"][()]
            self.Ntheta = file["parameters/Ntheta"][()]
            self.timestep = file["parameters/timestep"][()]
            self.kappa = file["parameters/kappa"][()]
            # forcing
            self.ellf = file["parameters/forcing/ellf"][()]
            self.sigmaf = file["parameters/forcing/sigmaf"][()]
            self.pf = file["parameters/forcing/pf"][()]
            # initial conditions
            self.ell0 = file["parameters/initial_conditions/ell0"][()]
            self.u0 = file["parameters/initial_conditions/u0"][()]
            self.i_kind = file["parameters/initial_conditions/i_kind"][()].decode()
            if self.i_kind == "segalini":
                self.sigmai = file["parameters/initial_conditions/sigmai"][()]


        # ---- initialise time ---- DONE

        self.time = 0


        # ---- build basis ---- DONE

        self.coords = d3.S2Coordinates('phi', 'theta')
        self.dist = d3.Distributor(self.coords, dtype=self.dtype)
        self.basis = d3.SphereBasis(self.coords, (self.Nphi, self.Ntheta), radius=self.R, dealias=1, dtype=self.dtype) # dealias = 1, since 3/2 is not needed here and it prevents shape conflicts


        # ---- build spectral grids ---- DONE

        # find global shape
        shape = self.basis.global_shape(grid_space=(False, False), scales=(1, 1))
        # find global dedalus spectral grid indices
        i_s = np.arange(shape[0])
        j_s = np.arange(shape[1])
        # create global spectral ij-grid
        i_grid, j_grid = np.meshgrid(i_s, j_s, indexing='ij')
        # find slicing to extract local spectral grid
        field = self.dist.Field(name='field', bases=self.basis)  # not really needed, as this is supposed to run on only 1 core
        local_slices = field.domain.dist.coeff_layout.slices(field.domain, scales=1) # not really needed, as this is supposed to run on only 1 core
        local_slice = local_slices[0] # not really needed, as this is supposed to run on only 1 core
        i_grid = i_grid[local_slice] # not really needed, as this is supposed to run on only 1 core
        j_grid = j_grid[local_slice] # not really needed, as this is supposed to run on only 1 core

        # spectral grid (grid indexing internally used by Dedalus)
        self.m_grid, self.ell_grid = self.basis.elements_to_groups((False, False), (i_grid, j_grid))
        # those grids save the l and m values on the spectral grid ordering provided by Dedalus

        # thinned mode grid (without spins)
        self.ell_grid_thinned = self.ell_grid[::2]
        self.m_grid_thinned = self.m_grid[::2]



        # ---- build real space grid ----

        phi = self.basis.local_grids(self.dist, scales=[1,1])[0] # real space units
        theta = self.basis.local_grids(self.dist, scales=[1,1])[1] # real space units

        # coordinate grid
        self.phi_grid, self.theta_grid = np.meshgrid(phi, theta, indexing='ij')
        # embedding in 3D cartesian space (for plotting)
        self.x_grid, self.y_grid, self.z_grid = pf.toCartesian(self.phi_grid, self.theta_grid)


        # some simplifications to calculate the structure functions

        # 1D vectors that list all grid point coordinates
        self.phis = self.phi_grid.flatten()
        self.thetas = self.theta_grid.flatten()

        # 3D vector, that list all grid points in cartesian coordinates
        # first entry is an array of all x-coordinates
        # second entry is an array of all y-coordinates
        # third entry is an array of all z-coordinates
        xs, ys, zs = pf.toCartesian(self.phis, self.thetas)
        self.xyzs = np.array([xs, ys, zs])     

        # 3D vectors of list of unit vector components at grid points in cartesian coordinates
        self.ephis = np.array([-np.sin(self.phis), np.cos(self.phis), np.zeros_like(self.phis)])
        self.ethetas = np.array([np.cos(self.thetas)*np.cos(self.phis), np.cos(self.thetas)*np.sin(self.phis), -np.sin(self.thetas)])




        # ---- initialise external fields ---- DONE

        # fields from raw data
        self.p = self.dist.Field(name='p', bases=self.basis) # pressure
        self.w = self.dist.Field(name='w', bases=self.basis) # vorticity
        self.wf = self.dist.Field(name='wf', bases=self.basis) # forcing vorticity

        # fields to be determined
        self.s = self.dist.Field(name='s', bases=self.basis) # velocity stream function
        self.sf = self.dist.Field(name='sf', bases=self.basis) # forcing stream function
        self.u = self.dist.VectorField(self.coords, name='u', bases=self.basis) # velocity
        self.f = self.dist.VectorField(self.coords, name='f', bases=self.basis) # forcing velocity

        # processing result fields
        self.E_field = np.zeros_like(self.ell_grid_thinned, dtype=self.dtype) # kinetic energy field
        self.Ef_field = np.zeros_like(self.ell_grid_thinned, dtype=self.dtype) # forcing energy field


        # ---- initialise Dedalus operators ---- DONE

        self.skewgrad_s = -d3.skew(d3.Gradient(self.s, self.coords)) # for calculation of velocity from stream function
        self.skewgrad_sf = -d3.skew(d3.Gradient(self.sf, self.coords)) # for calculation of forcing velocity from forcing stream function


        # ---- binning for spectra ---- DONE

        # -- bins -- 

        ell_max = np.max(self.ell_grid_thinned)

        if pp.ell_log_binning: # logarithmic binning
            # bin centers such that first center is 1
            num_bins = int(np.floor(np.log10(ell_max) * pp.ell_num_bins_per_decade)) + 1
            self.ell_bin_centers = 10**(np.arange(num_bins) / pp.ell_num_bins_per_decade)
            # construct inner bin boundaries
            self.ell_bins = np.zeros(num_bins+1)
            self.ell_bins[1:-1] = np.sqrt(self.ell_bin_centers[:-1] * self.ell_bin_centers[1:])
            # first and last boundary
            self.ell_bins[0] = self.ell_bin_centers[0]**2 / self.ell_bins[1]
            self.ell_bins[-1] = self.ell_bin_centers[-1]**2 / self.ell_bins[-2]
            # bin width vector
            self.ell_bin_width = np.diff(self.ell_bins)
            
        else: # linear binning
            # bin centers
            self.ell_bin_centers = np.linspace(0.0, ell_max, ell_max + 1) # bin centers = ell_values
            # bin boundaries
            self.ell_bins = np.zeros(len(self.ell_bin_centers) + 1)
            self.ell_bins[1:-1] = (self.ell_bin_centers[:-1] + self.ell_bin_centers[1:]) / 2
            self.ell_bins[0] = self.ell_bins[1] - 1 
            self.ell_bins[-1] = self.ell_bins[-2] + 1
            # bin width vector
            self.ell_bin_width = np.diff(self.ell_bins)


        # -- subpixel grid for logarithmic binning --
        # width of a subcell
        self.dell = 1 / pp.ell_subdivisions
        # computes subpixel offsets (centered in cell)
        self.sub_ell = np.arange(-0.5, 0.5, self.dell)
        ell_flat = self.ell_grid_thinned.reshape(-1, order='C')
        self.sub_ell_flat = (ell_flat[None, :] + self.sub_ell[:, None]).reshape(-1, order='C')



        # ---- binning for S3 ----

        # -- bins -- 

        if pp.l_log_binning: # logarithmic binning

            num_bins = int(np.floor(np.log10(self.Nphi/2) * pp.l_num_bins_per_decade)) + 1 # number of decades * number of bins per decade
            self.l_bin_centers = 2*np.pi/self.Nphi * 10**(np.arange(num_bins) / pp.l_num_bins_per_decade)

            # construct inner bin boundaries
            self.l_bins = np.zeros(num_bins+1)
            self.l_bins[1:-1] = np.sqrt(self.l_bin_centers[:-1] * self.l_bin_centers[1:])

            # first and last boundary
            self.l_bins[0] = self.l_bin_centers[0]**2 / self.l_bins[1]
            self.l_bins[-1] = self.l_bin_centers[-1]**2 / self.l_bins[-2]

            # bin width vector
            self.l_bin_width = np.diff(self.l_bins)

        else: # linear binning

            self.l_bins = np.linspace(0.0, np.pi*self.R, pp.l_num_bins + 1) # bin interval boundaries, creates an array of wave numbers for the binning
            self.l_bin_width = np.diff(self.l_bins) # width of a bin
            self.l_bin_centers = self.l_bins[:-1] + self.l_bin_width / 2 # bin centers (length = num_bins = length of self.q_bins - 1)



        # ---- theoretical average spectra ----

        # defines theoretical spectra (self.E_spectrum_init, self.F_spectrum_init, self.P_spectrum_init, self.Ef_spectrum_theo, self.Ff_spectrum_theo, self.Pf_spectrum_theo)
        self._theoretical_quantities() # uses the self.q_bins



 # _____ INTERNAL METHODS _____


    def _theoretical_quantities(self): # DONE

        '''
        function that calculates the theoretical quantities:
        initial conditions:
            - average initial energy spectrum
            - average initial enstrophy spectrum
            - average initial palinstrophy spectrum 
        forcing:
            - average forcing spectrum
            - average enstrophy spectrum
            - average palinstrophy spectrum        
        '''

        # ---- average initial energy spectrum ----
        # units: energy per wavenumber
        if self.i_kind == "herring":
            self.E_spectrum_init = (self.u0**2)*self.ell_bin_centers/(2*self.ell0**2)*np.exp(-self.ell_bin_centers/self.ell0)

        elif self.i_kind == "segalini":
            self.E_spectrum_init = self.u0**2/(2*self.sigmai*np.sqrt(2*np.pi)) * np.exp(-((self.ell_bin_centers-self.ell0)/(self.sigmai))**2/2)


        # ---- average initial enstrophy spectrum ----
        # units: enstrophy per wavenumber
        self.F_spectrum_init = self.ell_bin_centers*(self.ell_bin_centers + 1) / self.R**2 * self.E_spectrum_init

        # ---- average initial palinstrophy spectrum ----
        # units: palinstrophy per wavenumber
        self.P_spectrum_init = self.ell_bin_centers*(self.ell_bin_centers + 1) / self.R**2 * self.F_spectrum_init


        # ---- average forcing power spectrum ----
        # units: energy per time and wavenumber
        self.Ef_spectrum_theo = self.pf*self.ell_bin_centers / (self.sigmaf*np.sqrt(2*np.pi)*self.ellf) * np.exp(-((self.ell_bin_centers-self.ellf)/self.sigmaf)**2/2)

        # ---- average forcing enstrophy spectrum ----
        # units: enstrophy per time and wavenumber
        self.Ff_spectrum_theo = self.ell_bin_centers*(self.ell_bin_centers + 1) / self.R**2 * self.Ef_spectrum_theo

        # ---- average forcing palinstrophy spectrum ----
        # units: palinstrophy per time and wavenumber
        self.Pf_spectrum_theo = self.ell_bin_centers*(self.ell_bin_centers + 1) / self.R**2 * self.Ff_spectrum_theo


        # ---- average total forcing power ----
        # units: energy per time
        self.Ef_theo = self.pf

        # ---- average total forcing enstrophy ----
        # units: enstrophy per time
        self.Ff_theo = ((3+1/self.ellf)*self.sigmaf**2 + self.ellf*(self.ellf + 1))*self.pf

        # ---- average total forcing palinstrophy ----
        # units: palinstrophy per time
        self.Pf_theo = ((15*self.ellf+6)*self.sigmaf**4 + (10*self.ellf**3 + 12*self.ellf**2 + 3*self.ellf)*self.sigmaf**2 + self.ellf**5 + 2*self.ellf**4 + self.ellf**3) * self.pf/self.ellf

    


    # _____ EXTERNAL METHODS _____

    # ---- CALCULATIONS ----


    def find_fields(self): # DONE

        ''' function that calculates the stream function and the velocity from vorticity '''

        # ---- calculate stream function ----
        self.s['c'][self.ell_grid != 0] = self.w['c'][self.ell_grid != 0] / (self.ell_grid*(self.ell_grid + 1))[self.ell_grid != 0] # see Arbeitsgruppenpraktikum Eq. (49) (apart from ell = 0)
        self.s['c'][self.ell_grid == 0] = 0 # set stream function to zero if q = 0

        # ---- calculate velocity ----
        helper = self.skewgrad_s.evaluate()
        helper.change_scales(1)
        self.u['g'] = helper['g']        



    def find_forcing_fields(self): # DONE

        ''' function that calculates the forcing stream function and the forcing velocity from forcing vorticity '''

        # ---- calculate forcing stream function ----

        self.sf['c'][self.ell_grid != 0] = self.wf['c'][self.ell_grid != 0] / (self.ell_grid*(self.ell_grid + 1))[self.ell_grid != 0] # see Arbeitsgruppenpraktikum Eq. (49) (apart from ell = 0)
        self.sf['c'][self.ell_grid == 0] = 0 # set stream function to zero if q = 0

        # ---- calculate forcing velocity ----
        helper = self.skewgrad_sf.evaluate()
        helper.change_scales(1)
        self.f['g'] = helper['g']


    
    def find_E_field(self): # DONE

        '''
        calculates the energy field from the stream function
        The physical meaning is for m=0:
        E_field_{l,0} = l(l+1)/(2*R^2) * 2*pi * |psi^+_{l,0}|^2)
        and for m!=0:
        E_field_{l,m} = l(l+1)/(2*R^2) * pi * (|psi^+_{l,m}|^2 + |psi^-_{l,m}|^2)
        '''

        # abbreviations
        ell = self.ell_grid_thinned
        m = self.m_grid_thinned
        
        sc = self.s['c']
        sc2 = sc**2 # square each stream coefficient
        sc2_sum = np.zeros_like(ell)
        sc2_sum = sc2[::2] + sc2[1::2] # sum over spins (also valid for m=0, since there all spin down coefficients are zero by construction)
        self.E_field[m == 0] = ell[m == 0]*(ell[m == 0] + 1) / (2*self.R**2) * 2*np.pi * sc2_sum[m == 0] /(4*np.pi) # should be energy per unit area (Dedalus coefficients give the total energy, but we want to normalise per area)
        self.E_field[m != 0] = ell[m != 0]*(ell[m != 0] + 1) / (2*self.R**2) * np.pi * sc2_sum[m != 0] /(4*np.pi) # should be energy per unit area



    def find_Ef_field(self): # DONE

        '''
        calculates the energy field from the stream function
        The physical meaning is for m=0:
        Ef_field_{l,0} = l(l+1)/(2*R^2) * 2*pi * |fpsi^+_{l,0}|^2)
        and for m!=0:
        Ef_field_{l,m} = l(l+1)/(2*R^2) * pi * (|fpsi^+_{l,m}|^2 + |fpsi^-_{l,m}|^2)
        '''

        # abbreviations
        ell = self.ell_grid_thinned
        m = self.m_grid_thinned
        
        sfc = self.sf['c']
        sfc2 = sfc**2 # square each stream coefficient
        sfc2_sum = np.zeros_like(ell)
        sfc2_sum = sfc2[::2] + sfc2[1::2] # sum over spins (also valid for m=0, since there all spin down coefficients are zero by construction)
        self.Ef_field[m == 0] = ell[m == 0]*(ell[m == 0] + 1) / (2*self.R**2) * 2*np.pi * sfc2_sum[m == 0] /(4*np.pi) # should be energy per unit area
        self.Ef_field[m != 0] = ell[m != 0]*(ell[m != 0] + 1) / (2*self.R**2) * np.pi * sfc2_sum[m != 0] /(4*np.pi) # should be energy per unit area
        
    

    def find_spectra(self): # DONE

        '''
        Calculates the energy spectrum E(q) by binning self.E_field.

        Derives
        - energy spectrum E(q)
        - enstrophy spectrum F(q)
        - energy dissipation spectrum (2 kappa F(q))
        - palistrophy spectrum P(q)
        - enstrophy dissipation spectrum (2 kappa P(q))
        '''

        # ---- calculate energy spectrum ----

        self.E_spectrum = np.zeros_like(self.ell_bin_centers)

        if pp.ell_log_binning: # logarithmic binning

            # distribute energy field to subcells
            E_sub = np.tile(self.E_field.reshape(-1, order='C'), len(self.sub_ell)) * self.dell

            # histogram
            self.E_spectrum, _ = np.histogram(self.sub_ell_flat, bins=self.ell_bins, weights=E_sub)
            self.E_spectrum = self.E_spectrum / self.ell_bin_width

        else: # linear binning
            
            for ell in self.ell_bin_centers:
                self.E_spectrum[self.ell_bin_centers == ell] = self.E_field[self.ell_grid_thinned == ell].sum()


        # ---- calculate enstrophy spectrum ----

        # enstrophy spectrum
        # units: enstrophy / wavenumber
        self.F_spectrum = (self.ell_bin_centers)*(self.ell_bin_centers + 1) / self.R**2 * self.E_spectrum
        # energy dissipation spectrum
        # units: energy / (time * wavenumber)
        self.Ediss_spectrum = 2*self.kappa * ((self.ell_bin_centers)*(self.ell_bin_centers + 1) - 2) * self.E_spectrum # see Lindborg/nordmark


        # ---- calculate palistrophy spectrum ----

        # palistrophy spectrum
        # units: palistrophy / wavenumber
        self.P_spectrum = (self.ell_bin_centers)*(self.ell_bin_centers + 1) / self.R**2 * self.F_spectrum
        # enstrophy dissipation spectrum
        # units: enstrophy / (time * wavenumber)
        self.Fdiss_spectrum = 2*self.kappa * ((self.ell_bin_centers)*(self.ell_bin_centers + 1) - 2) * (self.ell_bin_centers)*(self.ell_bin_centers + 1) * self.E_spectrum



    def find_f_spectra(self): # DONE

        '''
        Calculates the forcing power spectrum Ef(q) by binning self.Ef_field.

        Derives
        - forcing energy spectrum Ef(q)
        - forcing enstrophy spectrum Ff(q)
        - forcing palistrophy spectrum Pf(q)
        '''
    
        # ---- calculate energy spectrum ----
        
        self.Ef_spectrum = np.zeros_like(self.ell_bin_centers)

        if pp.ell_log_binning: # logarithmic binning

            # distribute energy field to subcells
            Ef_sub = np.tile(self.Ef_field.reshape(-1, order='C'), len(self.sub_ell)) * self.dell

            # histogram
            self.Ef_spectrum, _ = np.histogram(self.sub_ell_flat, bins=self.ell_bins, weights=Ef_sub)
            self.Ef_spectrum = self.Ef_spectrum / self.ell_bin_width

        else: # linear binning
            
            for ell in self.ell_bin_centers:
                self.Ef_spectrum[self.ell_bin_centers == ell] = self.Ef_field[self.ell_grid_thinned == ell].sum()
 

        # ---- calculate enstrophy spectrum ----

        # forcing enstrophy spectrum
        # units: enstrophy / (time * wavenumber)
        self.Ff_spectrum = (self.ell_bin_centers)*(self.ell_bin_centers + 1) / self.R**2 * self.Ef_spectrum


        # ---- calculate palistrophy spectrum ----

        # forcing palistrophy spectrum
        # units: palistrophy / (time * wavenumber)
        self.Pf_spectrum = (self.ell_bin_centers)*(self.ell_bin_centers + 1) / self.R**2 * self.Ff_spectrum



    def find_total_quantities(self): # DONE

        '''
        Calculates the total energy, enstrophy, palistrophy and their dissipation.
        '''

        # ---- total quantities ----

        # total kinetic energy
        # units: energy
        self.E = np.sum(self.E_field)

        # total enstrophy
        # units: enstrophy
        self.F = np.sum((self.ell_grid_thinned)*(self.ell_grid_thinned + 1) * self.E_field)
        # total energy dissipation
        # units: energy / time
        self.Ediss = 2*self.kappa * np.sum((self.ell_grid_thinned*(self.ell_grid_thinned + 1) - 2) * self.E_field)

        # total palistrophy
        # units: palistrophy
        self.P = np.sum((self.ell_grid_thinned)**2 * (self.ell_grid_thinned + 1)**2 * self.E_field)
        # total enstrophy dissipation
        # units: enstrophy / time
        self.Fdiss = 2*self.kappa * np.sum((self.ell_grid_thinned*(self.ell_grid_thinned + 1) - 2) * self.ell_grid_thinned*(self.ell_grid_thinned + 1) * self.E_field)


    
    def find_total_f_quantities(self): # DONE

        '''
        Calculates the total forcing power, forcing enstrophy and forcing palistrophy.
        '''


        # ---- total forcing quantities ----

        # total forcing power
        # units: energy / time
        self.Ef = np.sum(self.Ef_field)

        # total forcing enstrophy
        # units: enstrophy / time
        self.Ff = np.sum((self.ell_grid_thinned)*(self.ell_grid_thinned + 1) * self.Ef_field)

        # total forcing palistrophy
        # units: palistrophy / time
        self.Pf = np.sum((self.ell_grid_thinned)**2 * (self.ell_grid_thinned + 1)**2 * self.Ef_field)



    def find_system_scales(self): # DONE

        '''
        Calculates multiple system scales such as the enstrophy dissipation scale.
        '''

        # ---- enstrophy dissipation scale ----

        self.ell_Fdiss = (self.Fdiss / (self.kappa)**3)**(1/6) # wave number of smallest structures (depends on enstrophy dissipation (and therefore on forcing) and on the viscosity)
        self.ell_Fdiss_theo = (self.Ff_theo / (self.kappa)**3)**(1/6) # theoretical wave number of smallest structures when enstrophy is conserved (depends on theoretical forcing)




    def find_Sp(self):

        # -- prepare parallelisaion --

        # total number of grid points
        size = len(self.phis)
        # number of pairs of points in one chunk
        # if too small, then the algorithm creates too many tasks and we have a lot of communication
        # if too large, the arrays of pairs of points per chunk become very large and need much memory
        chunk_size = 100_000

        # create chunks
        chunks = pf.create_chunks(size, chunk_size)

        
        # -- generate cartesian velocity vectors --

        # vectors of velocity components
        uphis = np.asarray(self.u['g'][0]).flatten()
        uthetas = np.asarray(self.u['g'][1]).flatten()

        # velocity in cartesian coordinates
        us = uphis*self.ephis + uthetas*self.ethetas


        # -- MAIN LOOP over all chunks --
        # is parallelised over the chunks

        start_Sp_time = time.time()

        # uses all available cores, uses threading (as dedalus needs MPI), pre_dispatch="2*n_jobs" means that 2*n_jobs jobs are created in advance, not all at once, this saves memory
        chunk_results = Parallel(n_jobs=-1, backend="loky", pre_dispatch="2*n_jobs")(delayed(pf.find_Sp_chunk)(chunk, us, self.xyzs, self.l_bins) for chunk in chunks)

        end_Sp_time = time.time()
        print("Time for find_Sp_loop: ", end_Sp_time - start_Sp_time)


        # -- combine results from all chunks --

        # simply add all results from all chunks
        # number of pairs of points in each bin
        Sp_num = np.sum(chunk_result[0] for chunk_result in chunk_results)
        # S2
        S2_sum = np.sum(chunk_result[1] for chunk_result in chunk_results)
        # S3
        S3_sum = np.sum(chunk_result[2] for chunk_result in chunk_results)


        # -- weight results by number of pairs of points per bin --

        self.S2 = np.where(Sp_num > 0, S2_sum/Sp_num, 0.0)
        self.S3 = np.where(Sp_num > 0, S3_sum/Sp_num, 0.0)


        # ---- calculate RHS of Kolmogorovs law ----
        
        # compare with analytical results for flat space and the sphere
        # - 3/2*epsilon*l (for energy cascade)
        self.S3_kolm_flat = - 3/2 * self.Ediss * self.l_bin_centers
        # - 1/8*eta*l^3 (for direct enstrophy cascade)
        self.S3_kolm_dir_flat = - 1/8 * self.Fdiss * self.l_bin_centers**3
        # beta*(epsilon*l)^{2/3} (for second order structure function in inertial subrange)
        self.S2_kolm_flat = self.Ediss**(2/3) * self.l_bin_centers**(2/3) # beta is ignored, because it is a constant and can be set such that it fits the graph while plotting
        
        # prediction from Bachelors thesis
        self.S3_kolm_curved = np.zeros_like(self.l_bin_centers)
        mask = self.l_bin_centers < np.pi/2

        self.S3_kolm_curved[mask] = -3*self.Ediss/2 * np.sin(self.l_bin_centers[mask]) * hyp2f1(1/2, 2, 3, np.sin(self.l_bin_centers[mask])**2)
        self.S3_kolm_curved[~mask] = 3*self.Ediss/2 * ( np.sin(self.l_bin_centers[~mask]) * hyp2f1(1/2, 2, 3, np.sin(self.l_bin_centers[~mask])**2) - 2/np.sin(self.l_bin_centers[~mask])**3 * hyp2f1(1/2, 2, 3, 1))


        # ---- calculate S2 from energy spectrum ----
        # cannot be done as I don not have developed the connection between spectral and real space for the sphere




    '''

    def find_Sp(self):

        # -- generate cartesian velocity vectors --

        # vectors of velocity components
        uphis = np.asarray(self.u['g'][0]).flatten()
        uthetas = np.asarray(self.u['g'][1]).flatten()

        # velocity in cartesian coordinates
        us = uphis*self.ephis + uthetas*self.ethetas


        # -- initializations --

        # dimensions of the grid
        size = len(self.phis)
        num_pairs = int(size*(size-1)/2)

        distances = np.zeros(num_pairs)
        du_parallels = np.zeros(num_pairs)

        # start loop counter
        counter = 0


        # -- loop --

        # outer loop over all grid points (points a)
        for i in range(size):

            #print(f'Calculating Sp for point {i}/{size}')

            a = self.xyzs[:, i]
            ua = us[:, i]


            # inner loop over all remaining grid points (points b)
            for j in range(i+1, size):

                b = self.xyzs[:, j]
                ub = us[:, j]


                # -- distance --

                dist = self._find_distance(a, b)

                # -- velocity increment --

                du_par = self._find_vel_increment(a, b, ua, ub)

                # -- fill saving vectors --

                distances[counter] = dist
                du_parallels[counter] = du_par


                # increase counter
                counter += 1


        # -- exclude nan values --

        # (intentionally created for antipodal points, when the algorithm becomes unstable)
        mask = ~np.isnan(distances) & ~np.isnan(du_parallels)
        distances = distances[mask]
        du_parallels = du_parallels[mask]


        # -- subpixel grid --
        # we skip the subpixel grid here
        # I think its not necessary, because there are much more distince distance values on the sphere than in flat space


        # -- calculate S3 and S2 (binning) --

        struc2 = du_parallels**2
        struc3 = du_parallels**3

        Sp_num, _ = np.histogram(distances, bins=self.l_bins)

        # S2
        S2_sum, _ = np.histogram(distances, bins=self.l_bins, weights=struc2) # sum over all S3 contributions in each bin 
        self.S2 = np.where(Sp_num > 0, S2_sum/Sp_num, 0.0)
        
        # S3
        S3_sum, _ = np.histogram(distances, bins=self.l_bins, weights=struc3) # sum over all S3 contributions in each bin 
        self.S3 = np.where(Sp_num > 0, S3_sum/Sp_num, 0.0)



        # ---- calculate RHS of Kolmogorovs law ----
        
        # compare with analytical results for flat space and the sphere
        # - 3/2*epsilon*l (for energy cascade)
        self.S3_kolm_flat = - 3/2 * self.Ediss * self.l_bin_centers
        # - 1/8*eta*l^3 (for direct enstrophy cascade)
        self.S3_kolm_dir_flat = - 1/8 * self.Fdiss * self.l_bin_centers**3
        # beta*(epsilon*l)^{2/3} (for second order structure function in inertial subrange)
        self.S2_kolm_flat = self.Ediss**(2/3) * self.l_bin_centers**(2/3) # beta is ignored, because it is a constant and can be set such that it fits the graph while plotting
        
        # prediction from Bachelors thesis
        self.S3_kolm_curved = np.zeros_like(self.l_bin_centers)
        mask = self.l_bin_centers < np.pi/2

        self.S3_kolm_curved[mask] = -3*self.Ediss/2 * np.sin(self.l_bin_centers[mask]) * hyp2f1(1/2, 2, 3, np.sin(self.l_bin_centers[mask])**2)
        self.S3_kolm_curved[~mask] = 3*self.Ediss/2 * ( np.sin(self.l_bin_centers[~mask]) * hyp2f1(1/2, 2, 3, np.sin(self.l_bin_centers[~mask])**2) - 2/np.sin(self.l_bin_centers[~mask])**3 * hyp2f1(1/2, 2, 3, 1))


        # ---- calculate S2 from energy spectrum ----
        # cannot be done as I don not have developed the connection between spectral and real space for the sphere

    '''


    def find_V(self): # TODO

        '''
        Calculates the real space energy spectrum V(l) from the second order structure function
        To reduce noise in the derivatives, the third order structure function is smoothed with a savgol filter.

        V(l) = -l^2/4 \partial_l [1/l \partial_l [S2]] # THIS MIGHT BE WRONG ?

        '''

        # ---- find V from S2 ----

        if pp.smooth:
            S2 = savgol_filter(self.S2, pp.filter_length, pp.filter_order)
        else:
            S2 = self.S2

        # create derivative operator using findiff
        diff = findiff.FinDiff(0, self.l_bin_width, 1, acc=pp.acc)
        # apply derivative
        help1 = diff(S2) 

        help2 = 1/self.l_bin_centers * help1

        help3 = diff(help2)

        self.V = -self.l_bin_centers**2 / 4 * help3



    
    def find_fluxes(self): # TODO

        # placeholder

        return 0




    # ---- SAVING ----


    def save_time(self, frame_index):

        '''
        Saves the time to the hdf5 file.
        name: always "time"
        frame_index: number of time (loop index)
        time: always fields.time
        '''

        with h5py.File(self.processed_path, "a") as f:

            grp = f.require_group("time")

            # add description for time
            if "description" not in grp.attrs:
                grp.attrs["description"] = "simulation time (allows to assign the time to each write number)"

            # if old data exists for this frame, delete
            if f"{frame_index:05d}" in grp:
                del grp[f"{frame_index:05d}"]

            # save time of this frame
            grp.create_dataset(f"{frame_index:05d}", data=self.time)



    def save_realfield(self, name, frame_index, realfield):

        '''
        Saves a real space field to the hdf5 file.
        name: e.g. "w"
        frame_index: number of frame (time)
        real field: e.g. fields.w['g']
        '''

        with h5py.File(self.processed_path, "a") as f:

            # access or create group 'spectra'
            grp = f.require_group("real_fields")

            # if not already present, add description to spectra group and create dataset for wavenumber bins 
            if "phi_grid" not in grp:
                grp.attrs["description"] = "all real space fields over the same real space grid"
                grp.create_dataset("phi_grid", data=self.phi_grid)
                grp.create_dataset("theta_grid", data=self.theta_grid)
                grp.create_dataset("x_grid", data=self.x_grid) # embedding in 3D cartesian space (for plotting)
                grp.create_dataset("y_grid", data=self.y_grid) # embedding in 3D cartesian space (for plotting)
                grp.create_dataset("z_grid", data=self.z_grid) # embedding in 3D cartesian space (for plotting)

            # if old data exists for this frame, delete
            if name + "/" + f"{frame_index:05d}" in grp:
                del grp[name + "/" + f"{frame_index:05d}"]

            # create dataset for the real space field
            grp.create_dataset(name + "/" + f"{frame_index:05d}", data=realfield)



    def save_spectralfield(self, name, frame_index, spectralfield):

        '''
        Saves a spectral field to the hdf5 file.
        name: e.g. "E_field"
        frame_index: number of frame (time)
        spectral field: e.g. fields.E_field
        '''

        with h5py.File(self.processed_path, "a") as f:

            # access or create group 'spectra'
            grp = f.require_group("spectral_fields")

            # if not already present, add description to spectra group and create dataset for spectral grid 
            if "ell_grid" not in grp:
                grp.attrs["description"] = "all spectral fields over the same spectral grid"
                grp.create_dataset("ell_grid", data=self.ell_grid_thinned)
                grp.create_dataset("m_grid", data=self.m_grid_thinned)

            # if old data exists for this frame, delete
            if name + "/" + f"{frame_index:05d}" in grp:
                del grp[name + "/" + f"{frame_index:05d}"]            

            # create dataset for the spectral field
            grp.create_dataset(name + "/" + f"{frame_index:05d}", data=spectralfield, dtype=self.dtype)



    def save_spectrum(self, name, frame_index, spectrum):

        '''
        Saves a spectrum to the hdf5 file.
        name: e.g. "Ef_spectrum"
        frame_index: number of frame (time)
        spectrum: e.g. fields.Ef_spectrum

        '''

        with h5py.File(self.processed_path, "a") as f:

            # access or create group 'spectra'
            grp = f.require_group("spectra")

            # if not already present, add description to spectra group and create dataset for wavenumber bins 
            if "modes" not in grp:
                grp.attrs["description"] = "all spectra over the same wavenumber bins"
                grp.create_dataset("modes", data=self.ell_bin_centers)
                grp.create_dataset("E_spectrum_init", data=self.E_spectrum_init)
                grp.create_dataset("F_spectrum_init", data=self.F_spectrum_init)
                grp.create_dataset("P_spectrum_init", data=self.P_spectrum_init)
                grp.create_dataset("Ef_spectrum_theo", data=self.Ef_spectrum_theo)
                grp.create_dataset("Ff_spectrum_theo", data=self.Ff_spectrum_theo)
                grp.create_dataset("Pf_spectrum_theo", data=self.Pf_spectrum_theo)

            # if old data exists for this frame, delete
            if name + "/" + f"{frame_index:05d}" in grp:
                del grp[name + "/" + f"{frame_index:05d}"]

            # create dataset for the spectrum
            grp.create_dataset(name + "/" + f"{frame_index:05d}", data=spectrum)


    
    def save_Sp(self, name, frame_index, Sp): # TODO
 
        '''
        Saves the structure function to the hdf5 file.
        name: "S3" or "S3_kolm"
        frame_index: number of frame (time)
        spectrum: fields.S3 or fields.S3_kolm

        '''

        with h5py.File(self.processed_path, "a") as f:

            # access or create group 'spectra'
            grp = f.require_group("structure_function")

            # if not already present, add description to S3 group and create dataset for bins 
            if "length" not in grp:
                grp.attrs["description"] = "parallel structure function of third and second order over length bins"
                grp.create_dataset("length", data=self.l_bin_centers)

            # if old data exists for this frame, delete
            if name + "/" + f"{frame_index:05d}" in grp:
                del grp[name + "/" + f"{frame_index:05d}"]

            grp.create_dataset(name + "/" + f"{frame_index:05d}", data=Sp)       





    def save_signature_and_fluxes(self, name, frame_index, flux): # TODO   

        # placeholder

        return 0


    
