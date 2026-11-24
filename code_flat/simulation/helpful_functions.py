'''
###################################################################################################################

HELPFUL FUNCTIONS

This file contains helpful functions for the simulation.

Overview:
1. Imports
2. Folder Management
    - create_run_folder

###################################################################################################################
'''



# __________ IMPORTS __________

import os # folder management
import h5py # for hdf5 data management
import json # for additional parameter file

import parameters as par # parameter file

from datetime import datetime # to store the date of the simulation



# __________ PARAMETER PROCESSING __________

# define timestep
if par.path_wise_forcing: # if pathwise forcing is activated
    timestep = par.a * par.timestep_min
else: # usually (without pathwise forcing)
    timestep = par.timestep




# __________ PARAMETER DICTIONARY __________

# defines which parameters are stored in the hdf5 file, also subdictionaries will transform to subgroups

initial_conditions = {
    "i_kind": str(par.i_kind),
    "u0": par.u0,
    "q0": par.q0,
    "i_seed": par.i_seed,
}

if par.i_kind == "segalini":
    initial_conditions["sigmai"] = par.sigmai

parameter_dict = {

    # COMPUTATIONAL PARAMETERS
    "simulation_time": par.simulation_time,
    "timestep": timestep,
    "save_time": par.save_time,
    "writes_per_file": par.writes_per_file,
    "path_wise_forcing": par.path_wise_forcing,
    "dtype": str(par.dtype),
    "dealias": par.dealias,

    # PHYSICAL PARAMETERS
    "N": par.N,
    "kappa": par.kappa,
    "initial_conditions": initial_conditions,
    "forcing": {
        "qf": par.qf,
        "sigmaf": par.sigmaf,
        "pf": par.pf,
        "f_seed": par.f_seed
    }
}




# __________ FOLDER MANAGEMENT __________


def save_dict_to_hdf5_group(group, d: dict):

    """
    Saves a (nested) dictionary recursively in an HDF5 group.
    Subdictionaries are converted to subgroups.
    """

    for key, value in d.items(): # key is e.g. "simulation_time", value is e.g. 100
        if isinstance(value, dict):
            # if the values is a subdictionary: create subgroups and fill recursively
            subgroup = group.create_group(key)
            save_dict_to_hdf5_group(subgroup, value)
        elif isinstance(value, str):
            # if the value is a string
            group.create_dataset(key, data=value, dtype=h5py.string_dtype())
        elif value is None:
            # if the value is None: cannot be saved directly → transform None to a string
            group.create_dataset(key, data="None", dtype=h5py.string_dtype())
        else:
            # if the value is a scalar
            group.create_dataset(key, data=value)



def create_run_folder(base_dir):

    '''
    creates a new folder in the base_directory for the current run,
    creates all subfolders (raw_data, processed_data, results)
    creates an hdf5 file in processed_data within it stores
        - metadata (run_id, description, date)
        - simulation parameters (from the above dictionary)
    '''

    # ---- create a directory for the current run ----

    # go to base directory
    os.makedirs(base_dir, exist_ok=True) 
    # extract numbers of existing run folders
    existing = [int(d.split("_")[1]) for d in os.listdir(base_dir) if d.startswith("run_")]
    # new run id = max run id + 1
    if existing == []:
        run_id = 1
    else:
        run_id = max(existing) + 1
    # create a run directory
    run_dir = os.path.join(base_dir, f"run_{run_id:05d}")
    os.makedirs(run_dir)


    # ---- create subfolders ----

    # create subfolders
    subfolders = ["raw_data", "processed_data", "results"] # one subfolder for raw data, one for processed data and one for results
    for sub in subfolders:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    # fill resultsubfolder with subsubfolders
    result_dir = os.path.join(run_dir, "results")
    resultfolders = ["vorticity", "forcing_vorticity", "pressure", "stream", "forcing_stream", "E_field", "forcing_E_field", "spectra", "balance_spectra", "evolution", "structure_function_2", "structure_function_3", "flux_V", "flux_E", "signature_function", "other", "videos"] # create folders to store results and plots
    for fold in resultfolders:
        os.makedirs(os.path.join(result_dir, fold), exist_ok=True)


    # ---- hdf5 file for parameters and processed data ----

    # create hdf5 file
    processed_file = os.path.join(run_dir, "processed_data", f"run_{run_id:05d}.h5") # create path to .h5 file
    with h5py.File(processed_file, "w") as f: # create .h5 file to store processed data
        
        # store METADATA
        f.attrs["run_id"] = run_id # run idendification number 
        f.attrs["description"] = par.description # description of the run
        f.attrs["date"] = datetime.now().isoformat() # date of the run

        # store PARAMETERS
        pg = f.create_group("parameters") # create a group to store the parameters
        pg.attrs["description"] = "simulation parameters from parameters.py"
        save_dict_to_hdf5_group(pg, parameter_dict) # saves the parameter dictionary as a group (see above def of save_dict_to_hdf5_group)


    # ---- output parameters in additional txt file ----
    # helpful for easier access

    with open(os.path.join(run_dir, "parameters.txt"), "w") as f:
        json.dump(parameter_dict, f, indent=4)


    # ---- return ----

    # return path of the folder for raw data and the run_id
    raw_folder = os.path.join(run_dir, "raw_data") #
    return (raw_folder, run_id)
    



