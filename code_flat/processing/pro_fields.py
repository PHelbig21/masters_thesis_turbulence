
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


        # ---- access hdf5 file ----

        self.processed_path = os.path.join(pp.run_dir, f"run_{pp.run_id:05d}", "processed_data", f"run_{pp.run_id:05d}.h5") # create path to .h5 file
    

        # ---- access simulation parameters ----
        
        # (stored in hdf5-file for processed data)
        self.par_path = os.path.join(pp.run_dir, "run_{:05d}/processed_data/run_{:05d}.h5".format(pp.run_id, pp.run_id))
        
        # read and store simulation parameters
        with h5py.File(self.par_path, mode='r') as file:

            # COMPUTATIONAL PARAMETERS
            dtype_str = file["parameters/dtype"][()].decode() # dtype_str = "<class 'numpy.float64'>"
            self.dtype = np.dtype(dtype_str.split("'")[1].split(".")[-1])
            self.dealias = file["parameters/dealias"][()]

            # PHYSICAL PARAMETERS
            self.L = 2*np.pi # hardcoded
            self.N = file["parameters/N"][()]
            self.timestep = file["parameters/timestep"][()]
            self.kappa = file["parameters/kappa"][()]
            # forcing
            self.qf = file["parameters/forcing/qf"][()]
            self.sigmaf = file["parameters/forcing/sigmaf"][()]
            self.pf = file["parameters/forcing/pf"][()]
            # initial conditions
            self.q0 = file["parameters/initial_conditions/q0"][()]
            self.u0 = file["parameters/initial_conditions/u0"][()]
            self.i_kind = file["parameters/initial_conditions/i_kind"][()].decode()
            if self.i_kind == "segalini":
                self.sigmai = file["parameters/initial_conditions/sigmai"][()]


        # ---- initialise time ----

        self.time = 0



        # ---- build basis ----   

        self.coords = d3.CartesianCoordinates('x', 'y')
        self.dist = d3.Distributor(self.coords, dtype=self.dtype)
        self.xbasis = d3.RealFourier(self.coords['x'], size=self.N, bounds=(0, self.L), dealias=1) # dealias = 1, since 3/2 is not needed here and it prevents shape conflicts
        self.ybasis = d3.RealFourier(self.coords['y'], size=self.N, bounds=(0, self.L), dealias=1) # dealias = 1, since 3/2 is not needed here and it prevents shape conflicts


        # ---- build spectral grids ----

        # spectral grid (grid indexing internally used by Dedalus)
        kx = self.dist.local_modes(self.xbasis)
        ky = self.dist.local_modes(self.ybasis)

        # wave number grid (grid indexing according to the physical mode number)
        self.qx = (kx//2)
        self.qy = (ky//2)
        qx_grid, qy_grid = np.meshgrid(self.qx, self.qy, indexing='ij') # fields of wave vector components
        self.q_grid = np.sqrt(qx_grid**2 + qy_grid**2) # field of absolute of wave vector

        # thinned wave number grid
        self.q_grid_thinned = self.q_grid[::2, ::2] # grid for only one coefficient field (instead of four)
        self.qx_grid_thinned = qx_grid[::2, ::2]
        self.qy_grid_thinned = qy_grid[::2, ::2] # self. davor, wenn es später in den Methoden gebraucht wird

        # weight for energy field
        weight_qx = np.where(self.qx_grid_thinned[:, 0] == 0, 1.0, 0.5)
        weight_qy = np.where(self.qy_grid_thinned[0, :] == 0, 1.0, 0.5)
        self.weight_thinned = weight_qx[:, None] * weight_qy[None, :]

        # truncation wave number
        self.trunc = min(np.max(self.qx), np.max(self.qy)) # needs to be adapted for MPI parallelisation (see field_factory)


        # ---- build real space grid ----

        x = self.dist.local_grid(self.xbasis) # real space units
        y = self.dist.local_grid(self.ybasis) # real space units

        # coordinate grid
        self.x_grid, self.y_grid = np.meshgrid(x, y, indexing='ij') # real space units
        # displacement grid (centererd coordinate grid, measures the displacement from the center, used for S3)
        self.lx_grid, self.ly_grid = np.meshgrid(x - x[int(self.N)//2, 0], y - y[0, int(self.N)//2], indexing='ij') # real space units


        # create a vector of all possible displacements
        # displacements in index units
        # boundaries must be set such that 0 is centered, otherwise the absolute correlation length is too large and does not respect the periodic boundaries
        self.displ_i = [
            (di, dj) # displacement vectors
            for di in range(-int(self.N)//2, int(self.N)//2) # displace in x-direction by -N/2 to N/2-1
            for dj in range(-int(self.N)//2, int(self.N)//2) # displace in y-direction by -N/2 to N/2-1
            if not (di == 0 and dj == 0) # exclude zero displacement as it gives zero anyway
        ]


        # ---- initialise external fields ----

        # fields from raw data
        self.p = self.dist.Field(name='p', bases=(self.xbasis, self.ybasis)) # pressure
        self.w = self.dist.Field(name='w', bases=(self.xbasis, self.ybasis)) # vorticity
        self.wf = self.dist.Field(name='wf', bases=(self.xbasis, self.ybasis)) # forcing vorticity

        # fields to be determined
        self.s = self.dist.Field(name='s', bases=(self.xbasis,self.ybasis)) # velocity stream function
        self.sf = self.dist.Field(name='sf', bases=(self.xbasis,self.ybasis)) # forcing stream function
        self.u = self.dist.VectorField(self.coords, name='u', bases=(self.xbasis, self.ybasis)) # velocity
        self.f = self.dist.VectorField(self.coords, name='f', bases=(self.xbasis, self.ybasis)) # forcing velocity

        # processing result fields
        self.E_field = np.zeros_like(self.q_grid_thinned, dtype=self.dtype) # kinetic energy field
        self.Ef_field = np.zeros_like(self.q_grid_thinned, dtype=self.dtype) # forcing energy field


        # ---- initialise Dedalus operators ----

        self.skewgrad_s = -d3.skew(d3.Gradient(self.s, self.coords)) # for calculation of velocity from stream function
        self.skewgrad_sf = -d3.skew(d3.Gradient(self.sf, self.coords)) # for calculation of forcing velocity from forcing stream function


        # ---- binning for spectra ----

        # -- bins --

        if pp.q_log_binning: # logarithmic binning
            
            # bin centers such that first center is 1
            num_bins = int(np.floor(np.log10(self.trunc) * pp.q_num_bins_per_decade)) + 1
            self.q_bin_centers = 10**(np.arange(num_bins) / pp.q_num_bins_per_decade)
            # construct inner bin boundaries
            self.q_bins = np.zeros(num_bins+1)
            self.q_bins[1:-1] = np.sqrt(self.q_bin_centers[:-1] * self.q_bin_centers[1:])
            # first and last boundary
            self.q_bins[0] = self.q_bin_centers[0]**2 / self.q_bins[1]
            self.q_bins[-1] = self.q_bin_centers[-1]**2 / self.q_bins[-2]
            # bin width vector
            self.q_bin_width = np.diff(self.q_bins)

        else: # linear binning
            
            self.q_bins = np.linspace(0.0, self.trunc, pp.q_num_bins + 1) # bin interval boundaries, creates an array of wave numbers for the binning
            self.q_bin_width = np.diff(self.q_bins) # width of bins
            self.q_bin_centers = self.q_bins[:-1] + self.q_bin_width / 2 # bin centers (length = num_bins = length of self.q_bins - 1)
        
        # -- subpixel grids --
        # width of a subcell
        self.dq = 1 / pp.q_subdivisions # (each cell is split into subdivisions subcells)
        # computes subpixel offsets (centered in cell)
        q_di, q_dj = np.meshgrid(np.arange(pp.q_subdivisions) - pp.q_subdivisions/2 + 0.5, np.arange(pp.q_subdivisions) - pp.q_subdivisions/2 + 0.5, indexing='ij') # creates subcell coordinates within a cell
        q_di, q_dj = q_di.flatten(), q_dj.flatten() # transform to 1D List
        # computes total subpixel grid (shape: (N, M, subdivisions^2))
        sub_qx = self.qx_grid_thinned[:, :, np.newaxis] + q_di[np.newaxis, np.newaxis, :] * self.dq # grid + subcell offset (index units) * subcell width
        sub_qy = self.qy_grid_thinned[:, :, np.newaxis] + q_dj[np.newaxis, np.newaxis, :] * self.dq # grid + subcell offset*cell width
        self.sub_q_flat = np.sqrt(sub_qx**2 + sub_qy**2).reshape(-1) # scalar wave number subpixel grid, flatten to 1D list with .reshape(-1)

        # -- truncation mask for subpixel grid --
        self.sub_q_mask = self.sub_q_flat < self.trunc # mask includes only values lower than the truncation wave number



        # ---- binning for S3 ----

        # -- bins --

        if pp.l_log_binning: # logarithmic binning
            
            # bin centers such that first center is 1
            num_bins = int(np.floor(np.log10(self.N/2) * pp.l_num_bins_per_decade)) + 1 # number of decades * number of bins per decade
            self.l_bin_centers = self.L/self.N * 10**(np.arange(num_bins) / pp.l_num_bins_per_decade)

            # construct inner bin boundaries
            self.l_bins = np.zeros(num_bins+1)
            self.l_bins[1:-1] = np.sqrt(self.l_bin_centers[:-1] * self.l_bin_centers[1:])

            # first and last boundary
            self.l_bins[0] = self.l_bin_centers[0]**2 / self.l_bins[1]
            self.l_bins[-1] = self.l_bin_centers[-1]**2 / self.l_bins[-2]

            # bin width vector
            self.l_bin_width = np.diff(self.l_bins)

        else: # linear binning

            self.l_bins = np.linspace(0.0, self.L/2, pp.l_num_bins + 1) # bin interval boundaries, creates an array of wave numbers for the binning
            self.l_bin_width = np.diff(self.l_bins) # width of bins
            self.l_bin_centers = self.l_bins[:-1] + self.l_bin_width / 2 # bin centers (length = num_bins = length of self.q_bins - 1)

        # -- subpixel grids --
        # width of a subcell
        # width of a cell / subdivisions
        self.dl = self.L / (self.N * pp.l_subdivisions) # real space units
        # computes subpixel offsets in index units
        l_di, l_dj = np.meshgrid(np.arange(pp.l_subdivisions) - pp.l_subdivisions/2 + 0.5, np.arange(pp.l_subdivisions) - pp.l_subdivisions/2 + 0.5, indexing='ij') # creates subcell coordinates within a cell
        l_di, l_dj = l_di.flatten(), l_dj.flatten() # transform to 1D List, shape (l_subdivisions^2)
        # computes total subpixel grid (shape: (N, N, subdivisions^2))
        sub_lx = self.lx_grid[:, :, np.newaxis] + l_di[np.newaxis, np.newaxis, :] * self.dl # subgrid = grid + subcell offset (index units) * subcell width
        sub_ly = self.ly_grid[:, :, np.newaxis] + l_dj[np.newaxis, np.newaxis, :] * self.dl # subgrid = grid + subcell offset (index units) * subcell width
        self.sub_l_flat = np.sqrt(sub_lx**2 + sub_ly**2).reshape(-1)  # shape (N^2 * l_subdivisions^2,)
        # -- truncation mask for subpixel grid --
        # exclude zero separation and beyond L/2
        self.sub_l_mask = (self.sub_l_flat > 0.0) & (self.sub_l_flat < self.L / 2)



        # ---- theoretical average spectra ----

        # defines theoretical spectra (self.E_spectrum_init, self.F_spectrum_init, self.P_spectrum_init, self.Ef_spectrum_theo, self.Ff_spectrum_theo, self.Pf_spectrum_theo)
        self._theoretical_quantities() # uses the self.q_bins




    # _____ INTERNAL METHODS _____


    def _theoretical_quantities(self):

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
            self.E_spectrum_init = (self.u0**2)*self.q_bin_centers/(2*self.q0**2)*np.exp(-self.q_bin_centers/self.q0)

        elif self.i_kind == "segalini":
            self.E_spectrum_init = self.u0**2/(2*self.sigmai*np.sqrt(2*np.pi)) * np.exp(-((self.q_bin_centers-self.q0)/(self.sigmai))**2/2)


        # ---- average initial enstrophy spectrum ----
        # units: enstrophy per wavenumber
        self.F_spectrum_init = self.q_bin_centers**2 * self.E_spectrum_init

        # ---- average initial palinstrophy spectrum ----
        # units: palinstrophy per wavenumber
        self.P_spectrum_init = self.q_bin_centers**2 * self.F_spectrum_init


        # ---- average forcing power spectrum ----
        # units: energy per time and wavenumber
        self.Ef_spectrum_theo = self.pf*self.q_bin_centers / (self.sigmaf*np.sqrt(2*np.pi)*self.qf) * np.exp(-((self.q_bin_centers-self.qf)/self.sigmaf)**2/2)

        # ---- average forcing enstrophy spectrum ----
        # units: enstrophy per time and wavenumber
        self.Ff_spectrum_theo = self.q_bin_centers**2 * self.Ef_spectrum_theo

        # ---- average forcing palinstrophy spectrum ----
        # units: palinstrophy per time and wavenumber
        self.Pf_spectrum_theo = self.q_bin_centers**2 * self.Ff_spectrum_theo


        # ---- average total forcing power ----
        # units: energy per time
        # self.Ef_theo = np.sum(self.Ef_spectrum_theo * self.q_bin_width)
        self.Ef_theo = self.pf

        # ---- average total forcing enstrophy ----
        # units: enstrophy per time
        # self.Ff_theo = np.sum(self.Ff_spectrum_theo * self.q_bin_width) 
        self.Ff_theo = (3*self.sigmaf**2+self.qf**2)*self.pf

        # ---- average total forcing palinstrophy ----
        # units: palinstrophy per time
        # self.Pf_theo = np.sum(self.Pf_spectrum_theo * self.q_bin_width)
        self.Pf_theo = (15*self.sigmaf**4 + 10*self.qf**2*self.sigmaf**2 + self.qf**4)*self.pf







    # _____ EXTERNAL METHODS _____

    # ---- CALCULATIONS ----


    def find_fields(self):

        ''' function that calculates the stream function and the velocity from vorticity '''

        # ---- calculate stream function ----
        self.s['c'][self.q_grid != 0] = self.w['c'][self.q_grid != 0] / (self.q_grid**2)[self.q_grid != 0] # divide by squared wave number (apart from q = 0)
        self.s['c'][self.q_grid == 0] = 0 # set stream function to zero if q = 0

        # ---- calculate velocity ----
        helper = self.skewgrad_s.evaluate()
        helper.change_scales(1)
        self.u['g'] = helper['g']


    
    def find_forcing_fields(self):

        ''' function that calculates the forcing stream function and the forcing velocity from forcing vorticity '''

        # ---- calculate forcing stream function ----

        self.sf['c'][self.q_grid != 0] = self.wf['c'][self.q_grid != 0] / (self.q_grid**2)[self.q_grid != 0] # divide by squared wave number (apart from q = 0)
        self.sf['c'][self.q_grid == 0] = 0 # set stream function to zero if q = 0
    
        # ---- calculate forcing velocity ----
        helper = self.skewgrad_sf.evaluate()
        helper.change_scales(1)
        self.f['g'] = helper['g']



    def find_E_field(self):

        '''
        calculates the energy field from the stream function more efficiently but according to
        self.E_field[i][j] = self.q_grid_thinned[i][j]**2/8*(sc[2*i][2*j]**2+sc[2*i][2*j+1]**2+sc[2*i+1][2*j]**2+sc[2*i+1][2*j+1]**2)
        The physical meaning is
        E_field = 1/2 * q^2 * |psi^_q|^2
        where
        |psi^_q|^2 = 1/4 * ((psi^++)^2 + (psi^+-)^2 + (psi^-+)^2 + (psi--)^2)
        '''
        
        sc = self.s['c']
        sc2 = sc.reshape(self.E_field.shape[0], 2, self.E_field.shape[1], 2) ** 2 # reshape into wave number blocs and square
        sc2_sum = sc2.sum(axis=(1, 3)) * self.weight_thinned  #/ 4  # Summe over squared values in 2x2 blocs, |psi^_q|^2 = 1/4 * ((psi^++)^2 + (psi^+-)^2 + (psi^-+)^2 + (psi--)^2) for inner modes and *2 for modes on axes (implemented by the weight)
        self.E_field  = self.q_grid_thinned**2 / 2 * sc2_sum   # calculate energy field: E_field = 1/2 * q^2 * |psi^_q|^2 



    def find_Ef_field(self):

        '''
        calculates the forcing energy field from the forcing stream function more efficiently but according to
        self.Ef_field[i][j] = self.q_grid_thinned[i][j]**2/8*(sfc[2*i][2*j]**2+sfc[2*i][2*j+1]**2+sfc[2*i+1][2*j]**2+sfc[2*i+1][2*j+1]**2)
        The physical meaning is
        Ef_field = 1/2 * q^2 * |psif^_q|^2
        where
        |psif^_q|^2 = 1/4 * ((psif^++)^2 + (psif^+-)^2 + (psif^-+)^2 + (psif--)^2)
        '''
        
        sfc = self.sf['c']
        sfc2 = sfc.reshape(self.Ef_field.shape[0], 2, self.Ef_field.shape[1], 2) ** 2 # reshape into wave number blocs and square
        sfc2_sum = sfc2.sum(axis=(1, 3)) * self.weight_thinned #/ 4  # Summe over squared values in 2x2 blocs, |psif^_q|^2 = 1/4 * ((psif^++)^2 + (psif^+-)^2 + (psif^-+)^2 + (psif--)^2)
        self.Ef_field  = self.q_grid_thinned**2 / 2 * sfc2_sum # calculate energy field: Ef_field = 1/2 * q^2 * |psif^_q|^2       



    def find_spectra(self):

        '''
        Calculates the energy spectrum E(q) by binning self.E_field into pp.num_bins bins.
        Each cell (i,j) is subdivided into (subdivisions x subdivisions) sub-cells for accuracy.

        Derives
        - energy spectrum E(q)
        - enstrophy spectrum F(q)
        - energy dissipation spectrum (2 kappa F(q))
        - palistrophy spectrum P(q)
        - enstrophy dissipation spectrum (2 kappa P(q))
        '''

        # ---- distribute energy ----

        # distribute energy from one cell to all subcells
        # energy per subcell = energy in cell / number of subcells = energy in cell * dq^2
        # repeat each value in flattend E_field subdivisions^2 times and split energy equally among subcells
        E_sub = np.repeat(self.E_field.reshape(-1), pp.q_subdivisions**2) * self.dq**2

    
        # ---- calculate energy spectrum ----

        # calculate energy spectrum by binning and radial summation
        # np.histogram(count subcells, within this bin, weight each count with the energy)
        # units: energy / wavenumber
        self.E_spectrum, _ = np.histogram(self.sub_q_flat[self.sub_q_mask], bins=self.q_bins, weights=E_sub[self.sub_q_mask])
        self.E_spectrum = self.E_spectrum / self.q_bin_width


        # ---- calculate enstrophy spectrum ----

        # enstrophy spectrum
        # units: enstrophy / wavenumber
        self.F_spectrum = self.q_bin_centers**2 * self.E_spectrum
        # energy dissipation spectrum (d_t E = - 2 kappa F)
        # units: energy / (time * wavenumber)
        self.Ediss_spectrum = 2*self.kappa * self.F_spectrum


        # ---- calculate palistrophy spectrum ----

        # palistrophy spectrum
        # units: palistrophy / wavenumber
        self.P_spectrum = self.q_bin_centers**2 * self.F_spectrum
        # enstrophy dissipation spectrum (d_t F = - 2 kappa P)
        # units: enstrophy / (time * wavenumber)
        self.Fdiss_spectrum = 2*self.kappa*self.P_spectrum 



    def find_f_spectra(self):

        '''
        Calculates the forcing power spectrum Ef(q) by binning self.Ef_field into pp.num_bins bins.
        Each cell (i,j) is subdivided into (subdivisions x subdivisions) sub-cells for accuracy.

        Derives
        - forcing energy spectrum Ef(q)
        - forcing enstrophy spectrum Ff(q)
        - forcing palistrophy spectrum Pf(q)
        '''

        # ---- distribute energy ----

        # distribute forcing energy from one cell to all subcells
        # energy per subcell = energy in cell / number of subcells = energy in cell * dq^2
        # repeat each value in flattend Ef_field subdivisions^2 times and split energy equally among subcells
        Ef_sub = np.repeat(self.Ef_field.reshape(-1), pp.q_subdivisions**2) * self.dq**2

    
        # ---- calculate energy spectrum ----
        
        # calculate energy spectrum by binning and radial summation
        # np.histogram(count subcells, within this bin, weight each count with the energy)
        # units: energy / (time * wavenumber)
        self.Ef_spectrum, _ = np.histogram(self.sub_q_flat[self.sub_q_mask], bins=self.q_bins, weights=Ef_sub[self.sub_q_mask])
        self.Ef_spectrum = self.Ef_spectrum / self.q_bin_width


        # ---- calculate enstrophy spectrum ----

        # forcing enstrophy spectrum
        # units: enstrophy / (time * wavenumber)
        self.Ff_spectrum = self.q_bin_centers**2 * self.Ef_spectrum


        # ---- calculate palistrophy spectrum ----

        # forcing palistrophy spectrum
        # units: palistrophy / (time * wavenumber)
        self.Pf_spectrum = self.q_bin_centers**2 * self.Ff_spectrum

    

    def find_total_quantities(self):

        '''
        Calculates the total energy, enstrophy, palistrophy and their dissipation.
        '''

        # ---- total quantities ----

        # total kinetic energy
        # units: energy
        #self.E = np.sum(self.E_spectrum) * self.q_bin_width
        self.E = np.sum(self.E_field)

        # total enstrophy
        # units: enstrophy
        self.F = np.sum(self.q_grid_thinned**2 * self.E_field)
        # total energy dissipation
        # units: energy / time
        self.Ediss = 2*self.kappa * self.F

        # total palistrophy
        # units: palistrophy
        self.P = np.sum(self.q_grid_thinned**4 * self.E_field)
        # total enstrophy dissipation
        # units: enstrophy / time
        self.Fdiss = 2*self.kappa * self.P



    def find_total_f_quantities(self):

        '''
        Calculates the total forcing power, forcing enstrophy and forcing palistrophy.
        '''


        # ---- total forcing quantities ----

        # total forcing power
        # units: energy / time
        self.Ef = np.sum(self.Ef_field)
        # total forcing enstrophy
        # units: enstrophy / time
        self.Ff = np.sum(self.q_grid_thinned**2 * self.Ef_field)

        # total forcing palistrophy
        # units: palistrophy / time
        self.Pf = np.sum(self.q_grid_thinned**4 * self.Ef_field)


    def find_system_scales(self):

        '''
        Calculates multiple system scales such as the enstrophy dissipation scale.
        '''

        # ---- enstrophy dissipation scale ----

        self.q_Fdiss = (self.Fdiss / (self.kappa)**3)**(1/6) # wave number of smallest structures (depends on enstrophy dissipation (and therefore on forcing) and on the viscosity)
        self.q_Fdiss_theo = (self.Ff_theo / (self.kappa)**3)**(1/6) # theoretical wave number of smallest structures when enstrophy is conserved (depends on theoretical forcing)




    def find_Sp(self):

        '''
        Calculates the S3 and S2 structure function from the velocity field self.u['g'].

        IDEA:
        Displace the domain versus itself using np.roll.
        Then substract both velocity fields to get a field of differences which must be projected onto the displacement vector.
        Do this for all possible displacements
        '''

        # ---- bring velocity field into useful shape ----
        u = np.transpose(self.u['g'], (1, 2, 0)) # shape (2, N, N) to (N, N, 2)


        # ---- build struc_field on displacement grid ----
        # struc3_field[ix, iy] stores S3 contribution for displacement (dx, dy)
        # struc2_field[ix, iy] stores S2 contribution for displacement (dx, dy)
        # index mapping: ix = dx + N//2  (so that dx=-N//2 -> ix=0)
        struc3_field = np.zeros((self.N, self.N))
        struc2_field = np.zeros((self.N, self.N))


        # ---- loop over all displacements ----

        start_Sp_time = time.time()

        # parallelised loop over all displacements 
        # parallelises the function self._calc_Sp
        # strucs has shape (di, dj, struc3, struc2)
        strucs = Parallel(n_jobs=-1, backend="loky")(delayed(pf.calc_Sp)(di, dj, u) for di, dj in self.displ_i)

        # save structure functions
        for di, dj, struc3, struc2 in strucs:

            ix = di + int(self.N)//2
            iy = dj + int(self.N)//2

            struc3_field[ix, iy] = struc3
            struc2_field[ix, iy] = struc2

        end_Sp_time = time.time()
        print("Time for find_Sp_loop: ", end_Sp_time - start_Sp_time)


        # ---- subpixel grid ----

        # subcells inherit the contribution of the cell they are in
        struc3_sub = np.repeat(struc3_field.reshape(-1), pp.l_subdivisions**2)
        struc2_sub = np.repeat(struc2_field.reshape(-1), pp.l_subdivisions**2)


        # ---- calculate S3 and S2 (binning) ----
        
        # calculate S3 and S2 by binning and radial summation
        # np.histogram(count subcells, within this bin, weight each count with the contribution)

        # number of subcells in each bin
        Sp_num, _ = np.histogram(self.sub_l_flat[self.sub_l_mask], bins=self.l_bins) # count the number of subcells in each bin

        # S3
        S3_sum, _ = np.histogram(self.sub_l_flat[self.sub_l_mask], bins=self.l_bins, weights=struc3_sub[self.sub_l_mask]) # sum over all S3 contributions in each bin  
        # mean value of S3 in each bin
        # mean value = sum of all S3 contributions in each bin / number of subcells in each bin
        self.S3 = np.where(Sp_num > 0, S3_sum / Sp_num, 0.0) # if no subcells in bin, do not divide by zero 

        # S2
        S2_sum, _ = np.histogram(self.sub_l_flat[self.sub_l_mask], bins=self.l_bins, weights=struc2_sub[self.sub_l_mask]) # sum over all S2 contributions in each bin  
        # mean value of S2 in each bin 
        # mean value = sum of all S2 contributions in each bin / number of subcells in each bin
        self.S2 = np.where(Sp_num > 0, S2_sum / Sp_num, 0.0) # if no subcells in bin, do not divide by zero


        # ---- calculate RHS of Kolmogorovs law ----
        # - 3/2*epsilon*l (for energy cascade)
        self.S3_kolm = - 3/2 * self.Ediss * self.l_bin_centers
        # - 1/8*eta*l^3 (for direct enstrophy cascade)
        self.S3_kolm_dir = - 1/8 * self.Fdiss * self.l_bin_centers**3
        # beta*(epsilon*l)^{2/3} (for second order structure function in inertial subrange)
        self.S2_kolm = self.Ediss**(2/3) * self.l_bin_centers**(2/3) # beta is ignored, because it is a constant and can be set such that it fits the graph while plotting
        # ~ eta^{2/3}l^2 (for direct cascade in second order structure function)
        self.S2_kolm_dir = self.Fdiss**(2/3) * self.l_bin_centers**2 # no prefactor known


        # ---- calculate S2 from energy spectrum ----
        # S2(l) = 2 \int_0^\infty E(q) (1-J_0(lq)) dq

        self.S2_from_spec = np.zeros(len(self.l_bin_centers))

        for i in range(len(self.l_bin_centers)):
            l = self.l_bin_centers[i]
            self.S2_from_spec[i] = np.sum(4 * self.E_spectrum * (0.5 - jv(1, l*self.q_bin_centers)/(l*self.q_bin_centers)) * self.q_bin_width) # for each l do hakel transform



    '''

    def find_Sp(self):

        
        #Calculates the S3 and S2 structure function from the velocity field self.u['g'].

        #IDEA:
        #Displace the domain versus itself using np.roll.
        #Then substract both velocity fields to get a field of differences which must be projected onto the displacement vector.
        #Do this for all possible displacements
        

        # ---- bring velocity field into useful shape ----
        u = np.transpose(self.u['g'], (1, 2, 0)) # shape (2, N, N) to (N, N, 2)

        # ---- build struc_field on displacement grid ----
        # struc3_field[ix, iy] stores S3 contribution for displacement (dx, dy)
        # struc2_field[ix, iy] stores S2 contribution for displacement (dx, dy)
        # index mapping: ix = dx + N//2  (so that dx=-N//2 -> ix=0)
        struc3_field = np.zeros((self.N, self.N))
        struc2_field = np.zeros((self.N, self.N))

        # ---- loop over all displacements ----

        start_Sp_time = time.time()

        for di, dj in self.displ_i:

            # -- get velocity difference vector field --

            u_shift = np.roll(u, (di, dj), axis=(0,1)) # shift plane by displacement vector
            du = u_shift - u # calculate field of velocity differences


            # -- projection onto displacement vector --

            # absolute displacement in index units
            dr_i = np.sqrt(di*di + dj*dj) # absolute displacement in index units
            # project plane onto displacement vector
            du_proj = du[:,:,0]*di/dr_i + du[:,:,1]*dj/dr_i # proj(u) = u * dr/|dr| = ux * dx/|dr| + uy * dy/|dr|


            # -- calculate structure field --
            
            # part of structure function
            struc3 = np.mean(du_proj**3) # averaged cubed parallel velocity difference
            struc2 = np.mean(du_proj**2) # averaged squared parallel velocity difference

            # -- store in displacement grid --
            ix = di + int(self.N)//2
            iy = dj + int(self.N)//2
            struc3_field[ix, iy] = struc3
            struc2_field[ix, iy] = struc2

        end_Sp_time = time.time()
        #print("Time for find_Sp_loop: ", end_Sp_time - start_Sp_time)

        # -- subpixel grid --

        # subcells inherit the contribution of the cell they are in
        struc3_sub = np.repeat(struc3_field.reshape(-1), pp.l_subdivisions**2)
        struc2_sub = np.repeat(struc2_field.reshape(-1), pp.l_subdivisions**2)

        # ---- calculate S3 and S2 (binning) ----
        
        # calculate S3 and S2 by binning and radial summation
        # np.histogram(count subcells, within this bin, weight each count with the contribution)

        # number of subcells in each bin
        Sp_num, _ = np.histogram(self.sub_l_flat[self.sub_l_mask], bins=self.l_bins) # count the number of subcells in each bin

        # S3
        S3_sum, _ = np.histogram(self.sub_l_flat[self.sub_l_mask], bins=self.l_bins, weights=struc3_sub[self.sub_l_mask]) # sum over all S3 contributions in each bin  
        # mean value of S3 in each bin
        # mean value = sum of all S3 contributions in each bin / number of subcells in each bin
        self.S3 = np.where(Sp_num > 0, S3_sum / Sp_num, 0.0) # if no subcells in bin, do not divide by zero 

        # S2
        S2_sum, _ = np.histogram(self.sub_l_flat[self.sub_l_mask], bins=self.l_bins, weights=struc2_sub[self.sub_l_mask]) # sum over all S2 contributions in each bin  
        # mean value of S2 in each bin 
        # mean value = sum of all S2 contributions in each bin / number of subcells in each bin
        self.S2 = np.where(Sp_num > 0, S2_sum / Sp_num, 0.0) # if no subcells in bin, do not divide by zero


        # ---- calculate RHS of Kolmogorovs law ----
        # - 3/2*epsilon*l (for energy cascade)
        self.S3_kolm = - 3/2 * self.Ediss * self.l_bin_centers
        # - 1/8*eta*l^3 (for direct enstrophy cascade)
        self.S3_kolm_dir = - 1/8 * self.Fdiss * self.l_bin_centers**3
        # beta*(epsilon*l)^{2/3} (for second order structure function in inertial subrange)
        self.S2_kolm = self.Ediss**(2/3) * self.l_bin_centers**(2/3) # beta is ignored, because it is a constant and can be set such that it fits the graph while plotting


        # ---- calculate S2 from energy spectrum ----
        # S2(l) = 2 \int_0^\infty E(q) (1-J_0(lq)) dq

        self.S2_from_spec = np.zeros(len(self.l_bin_centers))

        for i in range(len(self.l_bin_centers)):
            l = self.l_bin_centers[i]
            self.S2_from_spec[i] = np.sum(4 * self.E_spectrum * (0.5 - jv(1, l*self.q_bin_centers)/(l*self.q_bin_centers)) * self.q_bin_width) # for each l do hakel transform

        
    '''


    def find_V(self):

        '''
        Calculates the real space energy spectrum V(l) from the second order structure function and alternatively from the spectral kinetic energy E(q) via Hankel transform.
        To reduce noise in the derivatives, the third order structure function is smoothed with a savgol filter.

        V(l) = -l^2/4 \partial_l [1/l \partial_l [S2]]

        V(l) = 1/l * int_0^infty E(q) lq J_3(lq) dq

        '''

        # ---- find V prediction from Kolmogorov ----

        self.V_kolm = self.Ediss**(2/3) * self.l_bin_centers**(-1/3)
        self.V_kolm_dir = self.Fdiss**(2/3) * self.l_bin_centers

        # ---- find V from S2 ----

        if pp.smooth:
            S2 = savgol_filter(self.S2, pp.filter_length, pp.filter_order)
        else:
            S2 = self.S2

        # create derivative operator using findiff
        diff = findiff.FinDiff(0, self.l_bin_centers, 1, acc=pp.acc)
        # apply derivative
        help1 = diff(S2) 

        help2 = 1/self.l_bin_centers * help1

        help3 = diff(help2)

        self.V = -self.l_bin_centers**2 / 4 * help3

        '''
        help1 = np.zeros(len(self.l_bins)) # lives on l_bins, but not on l_bins[0] and l_bins[-1], there it is set to zero
        for i in range(len(self.l_bins)-2):
            help1[i+1] = (S2[i+1] - S2[i])/self.l_bin_width
        
        help2 = np.zeros(len(self.l_bins)) # lives on l_bins, but not on l_bins[0] and l_bins[-1], there it is set to zero
        help2[1:-1] = 1/self.l_bins[1:-1] * help1[1:-1]

        help3 = np.zeros(len(self.l_bin_centers)) # lives on l_bin_centers but not on l_bin_centers[0] and l_bin_centers[-1], there it is set to zero
        for i in range(len(self.l_bin_centers)-2):
            help3[i+1] = (help2[i+2] - help2[i+1])/self.l_bin_width

        self.V = -self.l_bin_centers[1:-1]**2 / 4 * help3[1:-1] # lives on l_bin_centers but Pi_V at l_bin_centers[0] and l_bin_centers[-1] is removed, so this vector is two values shorter, as this is necessary due do the calculation of two derivatives
        '''

        # ---- find V from E(q) ----

        self.V_hankel = np.zeros(len(self.l_bin_centers))

        for i in range(len(self.l_bin_centers)):

            l = self.l_bin_centers[i]

            integrand = self.E_spectrum * l * self.q_bin_centers * jv(3, l * self.q_bin_centers)
            integral = np.sum(integrand * self.q_bin_width)
            V = 1/l * integral

            self.V_hankel[i] = V





    def find_fluxes(self):
     
        '''
        Calculates the real space and spectral space energy fluxes from the third order structure function.
        To reduce noise in the derivatives, the third order structure function is smoothed with a savgol filter.
        The values at the boundaries are set to zero (no physical meaning, but necessary as we calculate derivatives by Euler method)

        Pi_V(l) = 1/12 * l^3 \partial_l[ 1/l^5 * \partial_l [ l^3 * S3 ] ]

        Pi_E(q) = -q^4/12 * \int_0^\infty [\partial_l [ l^3 * S3 ] * J_2(q*l)/(q*l)**2] dl

        T = 1/12 * q^3 \int_0^\infty [\partial_l [ l^3 * S3 ] * J_1(q*l)/(q*l)] dl
        '''

        # ---- find Pi_V ----

        if pp.smooth:
            S3 = savgol_filter(self.S3, pp.filter_length, pp.filter_order)
        else:
            S3 = self.S3

        help1 = self.l_bin_centers**3 * S3 # lives on l_bin_centers
        
        # create derivative operator using findiff
        diff = findiff.FinDiff(0, self.l_bin_centers, 1, acc=pp.acc)
        # apply derivative
        help2 = diff(help1)

        help3 = 1/(self.l_bin_centers**5) * help2

        help4 = diff(help3)

        self.Pi_V = 1/12 * self.l_bin_centers**3 * help4
        
        '''
        help2 = np.zeros(len(self.l_bins)) # lives on l_bins, but not on l_bins[0] and l_bins[-1], there it is set to zero
        for i in range(len(self.l_bins)-2):
            help2[i+1] = (help1[i+1] - help1[i])/self.l_bin_width

        help3 = np.zeros(len(self.l_bins)) # lives on l_bins, but not on l_bins[0] and l_bins[-1], there it is set to zero
        help3[1:-1] = 1/(self.l_bins[1:-1]**5) * help2[1:-1] # avoid division by zero

        help4 = np.zeros(len(self.l_bin_centers)) # lives on l_bin_centers but not on l_bin_centers[0] and l_bin_centers[-1], there it is set to zero
        for i in range(len(self.l_bin_centers)-2):
            help4[i+1] = (help3[i+2] - help3[i+1])/self.l_bin_width    

        self.Pi_V = 1/12 * self.l_bin_centers[1:-1]**3 * help4[1:-1] # lives on l_bin_centers but Pi_V at l_bin_centers[0] and l_bin_centers[-1] is removed, so this vector is two values shorter, as this is necessary due do the calculation of two derivatives
        '''

        # ---- find Pi_E ----

        self.Pi_E = np.zeros(len(self.q_bin_centers)) # lives on q_bin_centers

        for i in range(len(self.q_bin_centers)):

            q = self.q_bin_centers[i]

            J2 = jv(2, q*self.l_bin_centers) # Bessel function, lives on l_bins

            help5 = np.zeros(len(self.l_bin_centers)) # lives on l_bins, but not on l_bins[0] and l_bins[-1], there it is set to zero
            help5 = help2 * J2 / (q*self.l_bin_centers)**2 # avoid division by zero

            help6 = np.sum(help5 * self.l_bin_width) # integral over l_bins

            self.Pi_E[i] = - 1/12 * q**4 * help6 # lives on q_bin_centers


        # ---- find T ----

        self.T = np.zeros(len(self.q_bin_centers)) # lives on q_bin_centers

        for i in range(len(self.q_bin_centers)):

            q = self.q_bin_centers[i]

            J1 = jv(1, q*self.l_bin_centers) # Bessel function, lives on l_bins

            help5 = np.zeros(len(self.l_bin_centers)) # lives on l_bins, but not on l_bins[0] and l_bins[-1], there it is set to zero
            help5 = help2 * J1 / (q*self.l_bin_centers) # avoid division by zero

            help6 = np.sum(help5 * self.l_bin_width) # integral over l_bins

            self.T[i] = 1/12 * q**3 * help6 # lives on q_bin_centers


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
            if "x_grid" not in grp:
                grp.attrs["description"] = "all real space fields over the same real space grid"
                grp.create_dataset("x_grid", data=self.x_grid)
                grp.create_dataset("y_grid", data=self.y_grid)

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
            if "qx_grid" not in grp:
                grp.attrs["description"] = "all spectral fields over the same spectral grid"
                grp.create_dataset("qx_grid", data=self.qx_grid_thinned)
                grp.create_dataset("qy_grid", data=self.qy_grid_thinned)

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
            if "wavenumber" not in grp:
                grp.attrs["description"] = "all spectra over the same wavenumber bins"
                grp.create_dataset("wavenumber", data=self.q_bin_centers)
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

            

    def save_Sp(self, name, frame_index, Sp):

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



    def save_signature_and_fluxes(self, name, frame_index, flux):

        '''
        Saves real space and spectral space fluxes to the hdf5 file.
        name: "Pi_V" or "Pi_E"
        frame_index: number of frame (time)
        flux: fields.Pi_V or fields.Pi_E
        '''

        with h5py.File(self.processed_path, "a") as f:

            # access or create group 'spectra'
            grp = f.require_group("signature_and_fluxes")

            # if not already present, add description to S3 group and create dataset for bins 
            if "length" not in grp:
                grp.attrs["description"] = "signature function and real and spectral kinetic energy fluxes over length or wavenumber"
                grp.create_dataset("length", data=self.l_bin_centers)
                grp.create_dataset("wavenumber", data=self.q_bin_centers)

            # if old data exists for this frame, delete
            if name + "/" + f"{frame_index:05d}" in grp:
                del grp[name + "/" + f"{frame_index:05d}"]

            grp.create_dataset(name + "/" + f"{frame_index:05d}", data=flux)