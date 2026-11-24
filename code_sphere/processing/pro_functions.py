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
import numpy as np
import random



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





# ---- USEFUL FUNCTIONS ----


# spherical coords to cartesian coords
def toCartesian(phi, theta):
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return((x,y,z))






# ---- HELPER FOR find_Sp() ----

def find_vel_increment(a, b, ua, ub):

    '''
    function that is called in find_Sp() to calculate the longitudinal velocity difference between two points       
    '''

    # -- velocity increment --

    # normal vector
    n = np.cross(a, b) # cross product
    norm = np.linalg.norm(n)
    if norm < 1e-12: # safety check, since for almost antipodal points this is unstable
        return np.nan
    n = n / norm # normalize

    # tangent vector of geodesics
    ga = np.cross(n, a) # cross product
    gb = np.cross(n, b) # cross product

    # parallel velocity difference
    du_proj = np.dot(ub, gb) - np.dot(ua, ga)

    return du_proj



def find_distance(a, b):

    '''
    function that is called in find_Sp() to calculate the geodesic distance between two points
    a and b are points on the sphere in cartesian coordinates  
    might be unstable for small angles or almost antipodal points, but should be less heavy than haversine    
    '''

    # -- distance --

    cosangle = np.dot(a, b)
    cosangle = np.clip(cosangle, -1.0, 1.0) # safety against numerical errors
    angle = np.arccos(cosangle)
    dist = angle # if radius = 1
    
    return dist
    


def create_chunks(size, chunk_size):

    '''
    generator that is called in find_Sp() to create chunks of pairs of points i,j
    - i and j are the indices of the points in the grid, so we compare each point with each
    - size is the number of points in the grid
    - chunk_size is the number of pairs of i,j that are computed in one chunk
    '''

    chunk = []
    chunk_num = 0

    # loop over all pairs of points, each pair appears only once and not double (would be the case if we looped j over all possible values)
    for i in range(size): 
        for j in range(i + 1, size):

            # append a pair of points to the chunk if it is randomly chosen
            if random.random() < pp.pair_percentage:
                chunk.append((i, j))

                # if the chunk is full
                if len(chunk) == chunk_size:
                    # the chunk is yielded (returned), but the function is not stopped, only paused
                    yield chunk
                    # the chunk is reset and the loop continues
                    chunk = []
                    chunk_num += 1
                    #print(f"Chunk {chunk_num} of {int(size*(size-1)/2/chunk_size * pp.pair_percentage)} created.")

    # the loop finishes
    # if the last chunk contains at least one pair of points
    if chunk:
        # the last chunk is also returned
        yield chunk

    # so in the end (when this function is done), all pairs of points appear in exactly one chunk




def find_Sp_chunk(chunk, us, xyzs, l_bins):

    # -- prepare empty arrays --

    chunk_size = len(chunk)

    distances = np.zeros(chunk_size)
    du_parallels = np.zeros(chunk_size)  


    # -- start loop counter --
    counter = 0    


    # -- loop over all pairs of points in the chunk --

    for i, j in chunk:

        # get point a and its velocity
        a = xyzs[:, i]
        ua = us[:, i]

        # get point b and its velocity
        b = xyzs[:, j]
        ub = us[:, j]

        # distance
        dist = find_distance(a, b)

        # velocity increment
        du_par = find_vel_increment(a, b, ua, ub)

        # fill saving vectors
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
    # I think its not necessary, because there are much more distinct distance values on the sphere than in flat space

    # -- calculate S3 and S2 (binning) --

    struc2 = du_parallels**2
    struc3 = du_parallels**3

    Sp_num, _ = np.histogram(distances, bins=l_bins)

    # S2
    S2_sum, _ = np.histogram(distances, bins=l_bins, weights=struc2) # sum over all S3 contributions in each bin 
    
    # S3
    S3_sum, _ = np.histogram(distances, bins=l_bins, weights=struc3) # sum over all S3 contributions in each bin 

    # return number of pairs of points per bin, results for S2 and S3 in this chunk
    return Sp_num, S2_sum, S3_sum