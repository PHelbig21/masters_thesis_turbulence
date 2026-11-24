'''
###################################################################################################################

PARAMETERS

This file contains all custom parameters for the simulation.

Overview:
1. Imports
2. Run Management
3. Computational Parameters
    - Time
    - Analysis
    - Technical Settings
4. Physical Parameters
    - Resolution
    - Dynamics
    - Initial Conditions
    - Forcing

###################################################################################################################
'''




# __________ IMPORTS __________

import numpy as np



# __________ RUN MANAGEMENT __________

save_dir = '../runs_used' # folder in which run is saved (numbered automatically)
description = 'F6Le' # add description of run



# __________ COMPUTATIONAL PARAMETERS __________

# --- TIME ---

# SIMULATION TIME
simulation_time = 50

# TIMESTEPPING (USUAL)
# timestep for single simulations (without pathwise fixed forcing)
timestep = 0.000125

# TIMESTEPPING (WITH PATHWISE FORCING)
# timestep settings for simulations with pathwise fixed forcing, needed for strong convergence tests
a = 1 # INTEGER!!! timestep multplier (timestep = a*timestep_min)
timestep_min = 0.000125 # minimal timestep within the simulations to be compared = highest time resolution of forcing


# --- ANALYSIS ---

# SAVING INTERVAL
save_time = 0.1 # time intervall to save data

# DATA STORAGE
writes_per_file = 10 # max number of stored timesteps per file


# --- TECHNICAL SETTINGS ---

# !!! pathwise forcing and resolution independence cannot be used at the same time !!!
# Pathwise forcing itself is resolution independent, but the initial conditions of pathwise forcing are not. So we cannot do spatial and temporal convergenve tests at the same time!

# PATHWISE FORCING
# enables the creation of numerical forcing from a fixed higher resolved forcing path, needed for strong convergence tests
path_wise_forcing = False
# False = random forcing without higher resolution than necessary (recommended if no strong convergence test are done, since this is much faster)
# True = pathwise fixed forcing for strong convergence tests (slower)

# RESOLUTION INDEPENDENCE
# enables the creation of mode-wise reproducable initial conditions and forcing, needed for spatial convergence tests
resolution_independence = True
# False = random initial conditions and forcing that is not reproducable between simulations with different resolutions (recommended if no spatial convergence test are done, since this is much faster)
# True = mode-wise reproducable initial conditions and forcing (allows to do strong spatial convergence tests)

# DATA TYPE
dtype = np.float64 # data type for dedalus fields

# DEALIASING
dealias = 3/2 # dealiasing (3/2 for quadratic nonlinearities in the Navier-Stokes equations)



# __________ PHYSICAL PARAMETERS __________

# --- RESOLUTION ---

N = 1024 # MUST BE EVEN!!! number of grid points in each direction in real space


# --- DYNAMICS ---

kappa = 0.0008 # kinematic viscosity
# rho = 1 # mass density, do not change!!! HARDCODED


# --- INITIAL CONDITIONS ---

# kind of initial conditions
# i_kind = 'herring' uses those initial conditions defined in Herring et al. E(q) = u0^2*q/q0 * exp(-q/q0)
# i_kind = 'segalini' similar initial conditions as in Segalini & Lindborg E(q) = u0^2/(2*sigmai * sqrt(2pi) * exp(-((q-q0)/sigmai)^2/2)
i_kind = 'herring'

# sampled from an inital energy spectrum E(q)
u0 = 1.0 # mean total initial velocity (velocity amplitude)
q0 = 1.5 # determines location of peak in initial energy spectrum (the higher, the smaller the initial structures)
i_seed = 96 # seed for random number generator to create initial conditions
sigmai = 1. # standard deviation of initial energy spectrum (only used for i_kind = 'segalini')


# --- FORCING ---

# sampled from a Gaussian forcing spectrum with mean qf, standard deviation sigmaf and amplitude pf
qf = 10.0 # forcing wave number (wave number where most energy is inserted) # DO NOT SET TO ZERO, even for decaying turbulence, otherwise the simulation produces NANs!!!
sigmaf = 1. # standard deviation of forcing (witdh of forcing peak)
pf = 0.0 #0.00322 # total forcing power (integral over theoretical forcing spectrum), units: energy per time
# if pf = 0: decaying turbulence
# if pf != 0: forced turbulence
f_seed = 30 # seed for random number generator to create random forcing



