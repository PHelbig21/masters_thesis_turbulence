
'''
###################################################################################################################

PROCESSING FUNCTIONS

This file contains useful functions for the processing.

Overview:
1. Imports
2. Functions

###################################################################################################################
'''


# __________ IMPORTS __________


# external files
import pro_parameters as pp

# file management
import os
import glob
import re
import h5py

# calculations
import numpy as np



# __________ FUNCTIONS __________


# ---- FILE MANAGEMENT ---- 


def list_raw_data():

    """
    Returns list of (path, n_writes) tuples for all raw data files.
    """
    
    # -- paths to raw data files --
    pattern = os.path.join(pp.run_dir, f"run_{pp.run_id:05d}/raw_data/raw_data_s*.h5")
    files   = sorted(glob.glob(pattern), key=lambda f: int(re.search(r"s(\d+)\.h5$", f).group(1)))

    # function to count number of writes
    def n_writes(path):
        with h5py.File(path, "r") as f: # open file
            return len(f["scales/sim_time"]) # count and return number of writes

    # return vector, where return[:][0] = files and return[:][1] = numbers of writes
    return [(path, n_writes(path)) for path in files]




def find_needed(requested, dependencies):

    """
    Function to find all data that needs to be calculated from the requested data and its dependencies.
    Returns list of data that needs to be calculated to get the requested data.
    """
    
    needed = set() # empty set of all data that needs to be calculated, no duplicates

    def visit(node): # function that adds a variable and its dependencies to the set
        if node in needed: # if variable is already in set
            return # do nothing, because its dependencies are also already in set
        needed.add(node) # if not already in set, add variable
        for dep in dependencies.get(node, []): # find its dependencies
            visit(dep) # add all dependencies to needed and also their dependencies and so on

    for o in requested: # loop over all requested variables
        visit(o) # if not already existing, add variable and dependencies to needed           

    return needed






# ---- HELPER FOR find_Sp() ----


def calc_Sp(di, dj, u):

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

    return di, dj, struc3, struc2