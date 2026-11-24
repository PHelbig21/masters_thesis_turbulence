'''
###################################################################################################################

SIMULATION

This file contains the main simulation of the forced Navier-Stokes equations.

Navigate to Folder: 
    cd flat_space/code

Execution:
    mpiexec -n 8 python3 simulation/simulation.py
    mpiexec -n 4 python3 simulation/simulation.py
    mpiexec -n 1 python3 simulation/simulation.py

Overview:
1. Imports
2. Parameter Processing
3. Data Management
    - Folder Management
    - Parameter Storage
4. Field Factory
    - Initialisation of the field factory implemented in field_factory.py
5. Initial Conditions
    - Creation of initial conditions and a forcing field using the field factory
6. Problem and Solver
    - Definition of the problem and solver
    - Analysis tools
    - Time stepping

###################################################################################################################
'''


# __________ IMPORTS __________


# external files
import parameters as par # simulation parameter file
import helpful_functions as func # file with helpful functions
import field_factory as ff # to create initial conditions and forcing

# Dedalus
import dedalus.public as d3

# parallelisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# performance
import time # time measurement

# logging
import logging
logger = logging.getLogger(__name__)

# calculation
import numpy as np




# __________ DATA MANAGEMENT __________


# --- CREATE STORAGE FOLDERS, STORE PARAMETERS ---

# let the master core create the run folder and save the path
if rank == 0: # first core creates the run folder and saves the path
    # create run folder with subfolders and return the path of the folder to store snapshots (raw data) and the run_id
    # func.create_run_folder also saves the metadata and parameters of the run in the postprocessing hdf5 file
    snapshot_folder, run_id = func.create_run_folder(par.save_dir)
else: # other cores do nothing
    snapshot_folder = None
    run_id = None

# share the data path information with all cores
snapshot_folder = comm.bcast(snapshot_folder, root=0)
run_id = comm.bcast(run_id, root=0)



# __________ SAFETY WARNINGS __________


if par.resolution_independence and par.path_wise_forcing:
    logger.warning("WARNING: Pathwise forcing and resolution independence cannot be used at the same time!")




# __________ FIELD FACTORY __________

# initialise a FieldFactory for bases, fields, field information, initial conditions and forcing
factory = ff.FieldFactory(par.i_seed, par.f_seed) # depends only on the seeds




# __________ INITIAL CONDITIONS __________

# ---- initial velocity and pressure field ----

# create a random initial velocity field, depending on the seed
u, p, tau_p = factory.create_initial_velocity() # see initial_conditions.py


# ---- initial forcing field ----

# creates a random initial forcing field ()
f = factory.create_forcing()




# __________ PROBLEM AND SOLVER __________


# ---- Problem ----

problem = d3.IVP([u, p, tau_p], namespace={'u': u, 'p': p, 'tau_p': tau_p, 'f': f, 'kappa': par.kappa})
problem.add_equation("dt(u) - kappa*lap(u) + grad(p) = - u@grad(u) + f") # Navier-Stokes
problem.add_equation("div(u) + tau_p = 0") # Continuity
problem.add_equation("integ(p) = 0") # pressure gauge (integ instead of ave, because ave does not work)

# Solver
solver = problem.build_solver(d3.RK222) #d3.RK222
solver.stop_sim_time = par.simulation_time

# Analysis
snapshots = solver.evaluator.add_file_handler(snapshot_folder, sim_dt=par.save_time, max_writes=par.writes_per_file) # e.g. save data every 0.1s, max 10 saves per file -> (20s) = 20 files
snapshots.add_task(p, name='pressure', layout='g')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity', layout='g')
snapshots.add_task(-d3.div(d3.skew(f))*np.sqrt(func.timestep), name='forcing_vorticity', layout='g') # save analytical forcing field (without rescaling with timestep)

# Solve
start_solving = time.time()

while solver.proceed:

    # refresh forcing term
    f.change_scales(1)
    f['g'] = factory.create_forcing()['g'] # creates numerical forcing (already rescaled with timestep)
    
    # timestep
    solver.step(func.timestep)

    # info output
    if (solver.iteration) % 100 == 0: # every 100th iteration:
        logger.info(f'Time={solver.sim_time:.2f}/{par.simulation_time}')

end_solving = time.time()

# Info on run and output
if rank == 0: # only on the master core
    print(f"Run {run_id} finished within {end_solving - start_solving:.4f} seconds.")
    print(f"Results can be found in {snapshot_folder}.")






