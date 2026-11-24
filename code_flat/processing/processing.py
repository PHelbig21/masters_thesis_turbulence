
'''
###################################################################################################################

PROCESSING

This file contains the main post-processing after simulation.py was executed.
The main computation time is spent on the calculation of the structure function.

Navigate to Folder: 
    cd flat_space/code
    cd runs_test/run_00009/processed_data

Execution:
    python3 processing/processing.py

Overview:
1. Imports
2. Preparation
3. Main Loop

###################################################################################################################
'''




# __________ IMPORTS __________

# external files
import pro_fields as pf
import pro_functions as func
import pro_parameters as pp

# file management
import h5py

# calculation
import numpy as np

# time measurement
from datetime import datetime
import sys
sys.stdout.reconfigure(line_buffering=True) # deactivate buffer (all print commands are immediately printed)
import time # time measurement

print(f'Processing starts at {datetime.now()}')


# __________ PREPARATION __________

start_processing = time.time()


# ---- dependencies ----
# decides what needs to be calculated to process all data requested in pro_parameters

# define which data needs which to be calculated
dependencies = {
    "fields": [],
    "forcing_fields": [],
    "E_field": ["fields"],
    "Ef_field": ["forcing_fields"],
    "spectra": ["E_field"],
    "f_spectra": ["Ef_field"],
    "total_quantities_evol": ["spectra"],
    "total_f_quantities_evol": ["f_spectra"],
    "system_scales": ["total_quantities_evol"],
    "fields_evol": ["fields"],
    "forcing_fields_evol": ["forcing_fields"],
    "Sp": ["fields", "total_quantities_evol"],
    "fluxes": ["Sp"],
    "signature_function": ["Sp"]
}

# collect all data that is requested
requested = {k for k, v in pp.requested_data.items() if v}
print(requested)

# find all data that needs to be calculated
needed = func.find_needed(requested, dependencies)
print(needed)


# ---- field class ----

# Create a class ProcessingFields and thereby create all spectral grids and Dedalus fields within this class
fields = pf.ProcessingFields() # this class reads the simulation parameters from the hdf5 file and reconstructs the setting

# ---- raw data files ----

# list raw data files and count number of writes per file
raw_files = func.list_raw_data() # raw_files[:][0] = files, raw_files[:][1] = numbers of writes
loops = np.sum([entry[1] for entry in raw_files]) # total number of loops and frames (used for printing progress)


# ---- prepare loop number ----

loop_num = 0


# ---- prepare time evolution arrays ----

# time
time_evol = []

# total quantities
E_evol = []
F_evol = []
Ediss_evol = []
P_evol = []
Fdiss_evol = []

# total forcing quantities
Ef_evol = []
Ff_evol = []
Pf_evol = []

# theoretical total forcing quantities
Ef_evol_theo = []
Ff_evol_theo = []
Pf_evol_theo = []

# system scales
q_Fdiss_evol = []
q_Fdiss_evol_theo = []

# standard deviations of real space fields
std_p_evol = []
std_w_evol = []
std_wf_evol = []
std_s_evol = []
std_sf_evol = []
std_ux_evol = []
std_uy_evol = []
std_fx_evol = []
std_fy_evol = []


# __________ MAIN LOOP __________
# loop over all frames 

# ---- loop over all raw data files ----

for file_num in range(len(raw_files)): 

    # -- read file data --

    with h5py.File(raw_files[file_num][0], mode='r') as file:

        p_set = file['tasks']['pressure'] # all pressure data stored in this file
        w_set = file['tasks']['vorticity'] # all vorticity data stored in this file
        wf_set = file['tasks']['forcing_vorticity'] # all forcing vorticity data stored in this file

        time_set = file['scales']['sim_time'] # all simulation times stored in this file



        # ---- loop over all writes within the file ----
        #for write_num in range(raw_files[file_num][1]): 

    # -- read file data --

    with h5py.File(raw_files[file_num][0], mode='r') as file:

        p_set = file['tasks']['pressure'] # all pressure data stored in this file
        w_set = file['tasks']['vorticity'] # all vorticity data stored in this file
        wf_set = file['tasks']['forcing_vorticity'] # all forcing vorticity data stored in this file

        time_set = file['scales']['sim_time'] # all simulation times stored in this file


        # ---- loop over all writes within the file ----
        #for write_num in range(raw_files[file_num][1]):
        
        for write_num in range(raw_files[file_num][1]):

            start_pro_time = time.time()

            # print(f"file_num: {file_num}")
            # print(f"write_num: {write_num}")
            print(f"Processing frame {loop_num}/{loops}")

            # -- read data for this write and pass it to the fields class --

            # read fields
            fields.p['g'] = p_set[write_num, :, :] # pressure field for this write
            fields.w['g'] = w_set[write_num, :, :] # vorticity field for this write
            fields.wf['g'] = wf_set[write_num, :, :] # forcing vorticity field for this write
            # save fields
            fields.save_realfield("p", loop_num, fields.p['g']) # save pressure
            fields.save_realfield("w", loop_num, fields.w['g']) # save vorticity
            fields.save_realfield("wf", loop_num, fields.wf['g']) # save vorticity

            # find standard deviation of fields (needed for scaling the plots of the real space fields)
            std_p_evol.append(np.std(fields.p['g'])) # save standard deviation of pressure
            std_w_evol.append(np.std(fields.w['g'])) # save standard deviation of vorticity
            std_wf_evol.append(np.std(fields.wf['g'])) # save standard deviation of forcing vorticity

            # read time
            fields.time = time_set[write_num] # simulation time of this write
            # save time
            time_evol.append(fields.time) # for the evolution plots
            fields.save_time(loop_num) # for assigning the time to each write


            # -- calculate stream and velocity fields --

            if "fields" in needed: # if fields are needed
                # w → s → u 
                fields.find_fields() # calculates stream function and velocity field from fields.w (vorticity) and stores it in fields.s and fields.u
            
            if "forcing_fields" in needed: # if forcing fields are needed
                # wf → sf → f
                fields.find_forcing_fields() # calculates forcing stream function and forcing field from fields.wf (forcing vorticity) and stores it in fields.sf and fields.f

            # save fields if they should be saved
            if "fields" in requested: # if fields are requested
                fields.save_realfield("s", loop_num, fields.s['g']) # save stream function
                fields.save_realfield("u", loop_num, fields.u['g']) # save velocity
            if "forcing_fields" in requested: # if forcing fields are requested
                fields.save_realfield("sf", loop_num, fields.sf['g']) # save forcing stream function
                fields.save_realfield("f", loop_num, fields.f['g']) # save forcing field

            # find standard deviation of fields (needed for scaling the plots of the real space fields)
            if "fields_evol" in needed: # if standard deviations of fields are needed
                std_s_evol.append(np.std(fields.s['g'])) # save standard deviation of pressure
                std_ux_evol.append(np.std(fields.u['g'][0])) # save standard deviation of the x-component of the velocity
                std_uy_evol.append(np.std(fields.u['g'][1])) # save standard deviation of the y-component of the velocity
            if "forcing_fields_evol" in needed: # if standard deviations of forcing fields are needed
                std_sf_evol.append(np.std(fields.sf['g'])) # save standard deviation of vorticity
                std_fx_evol.append(np.std(fields.f['g'][0])) # save standard deviation of the x-component of the forcing velocity
                std_fy_evol.append(np.std(fields.f['g'][1])) # save standard deviation of the y-component of the forcing velocity


            # -- calculate energy fields -- 

            if "E_field" in needed: # if E_field is needed
                # s → E_field
                fields.find_E_field() # uses fields.w (vorticity fields) to find fields.E_field
            if "Ef_field" in needed: # if Ef_field is needed
                # sf → Ef_field
                fields.find_Ef_field() # uses fields.wf (forcing vorticity field) to find fields.Ef_field

            if "E_field" in requested: # if E_field is requested:
                # save E_field
                fields.save_spectralfield("E_field", loop_num, fields.E_field)
            if "Ef_field" in requested: # if Ef_field is requested
                # save Ef_field
                fields.save_spectralfield("Ef_field", loop_num, fields.Ef_field)


            # -- calculate spectra --

            if "spectra" in needed: # if spectra are needed
                # E_field → E_spectrum → F_spectrum, Ediss_spectrum → P_spectrum, Fdiss_spectrum
                fields.find_spectra() # uses fields.E_field to find fields.E_spectrum, fields.P_spectrum, fields.Ediss_spectrum, fields.P_spectrum, fields.Fdiss_spectrum
            if "f_spectra" in needed: # if forcing spectra are needed
                # Ef_field → Ef_spectrum → Ff_spectrum → Pf_spectrum
                fields.find_f_spectra() # uses fields.Ef_field to find fields.Ef_spectrum, fields.Ff_spectrum, fields.Pf_spectrum

            if "spectra" in requested: # if spectra are requested
                # save all spectra
                fields.save_spectrum("E_spectrum", loop_num, fields.E_spectrum)
                fields.save_spectrum("F_spectrum", loop_num, fields.F_spectrum)
                fields.save_spectrum("Ediss_spectrum", loop_num, fields.Ediss_spectrum)
                fields.save_spectrum("P_spectrum", loop_num, fields.P_spectrum)
                fields.save_spectrum("Fdiss_spectrum", loop_num, fields.Fdiss_spectrum)
            if "f_spectra" in requested: # if forcing spectra are requested
                # save all f spectra
                fields.save_spectrum("Ef_spectrum", loop_num, fields.Ef_spectrum)
                fields.save_spectrum("Ff_spectrum", loop_num, fields.Ff_spectrum)
                fields.save_spectrum("Pf_spectrum", loop_num, fields.Pf_spectrum)


            # -- calculate total quantities --

            if "total_quantities_evol" in needed: # if total quantities are needed:
                # spectra → total values
                fields.find_total_quantities() # calculates total quantities (total kinetic energy, total enstrophy, total palistrophy and dissipations)
                # store total quantities
                E_evol.append(fields.E)
                F_evol.append(fields.F)
                Ediss_evol.append(fields.Ediss)
                P_evol.append(fields.P)
                Fdiss_evol.append(fields.Fdiss)

            if "total_f_quantities_evol" in needed: # if total forcing quantities are needed
                # f spectra → total f values
                fields.find_total_f_quantities() # calculates total forcing quantities (total forcing power, total forcing enstrophy, total forcing palistrophy)
                # store total forcing quantities
                Ef_evol.append(fields.Ef)
                Ff_evol.append(fields.Ff)
                Pf_evol.append(fields.Pf)
                # store theoretical total forcing quantities (calculate in the initialisation, not here)
                Ef_evol_theo.append(fields.Ef_theo)
                Ff_evol_theo.append(fields.Ff_theo)
                Pf_evol_theo.append(fields.Pf_theo)


            # -- calculate system scales --

            if "system_scales" in needed: # if system scales are needed
                # total quantities → system scales 
                fields.find_system_scales() # calculates system scales such as the enstrophy dissipation scale
                q_Fdiss_evol.append(fields.q_Fdiss)
                q_Fdiss_evol_theo.append(fields.q_Fdiss_theo)


            # -- calculate structure function --
            # not all frames might be processed to save computing time

            if "Sp" in needed: # if structure functions are needed

                if isinstance(pp.Sp_frames, str) and pp.Sp_frames == "all": # if all structure function frames should be calculated

                    print(f'Calculating Sp for frame {loop_num}/{loops}')
                    start_find_Sp_time = time.time()

                    fields.find_Sp() # find structure functions

                    stop_find_Sp_time = time.time()
                    #print('Time for finding Sp: ', stop_find_Sp_time - start_find_Sp_time)

                    if "Sp" in requested: # if structure functions are requested
                        fields.save_Sp("S3", loop_num, fields.S3) # save third order structure function
                        fields.save_Sp("S2", loop_num, fields.S2) # save second order structure function
                        fields.save_Sp("S3_kolm", loop_num, fields.S3_kolm) # save theoretical S3 for energy cascade
                        fields.save_Sp("S3_kolm_dir", loop_num, fields.S3_kolm_dir) # save theoretical S3 for direct enstrophy cascade
                        fields.save_Sp("S2_kolm", loop_num, fields.S2_kolm) # save theoretical S2 for second order structure function
                        fields.save_Sp("S2_kolm_dir", loop_num, fields.S2_kolm_dir) # save theoretical S2 for second order structure function for direct cascade
                        fields.save_Sp("S2_from_spec", loop_num, fields.S2_from_spec) # save S2 calculated via hankel transform from spectrum

                    print(f'Calculating Sp done')

                else: # if only specific frames should be processed

                    if loop_num in pp.Sp_frames: # if the loop number corresponds to a frame that should be processed

                        print(f'Calculating Sp for frame {loop_num}/{loops}')
                        fields.find_Sp()

                        if "Sp" in requested: # if structure functions are requested
                            fields.save_Sp("S3", loop_num, fields.S3) # save third order structure function
                            fields.save_Sp("S2", loop_num, fields.S2) # save second order structure function
                            fields.save_Sp("S3_kolm", loop_num, fields.S3_kolm) # save theoretical S3 for energy cascade
                            fields.save_Sp("S3_kolm_dir", loop_num, fields.S3_kolm_dir) # save theoretical S3 for direct enstrophy cascade
                            fields.save_Sp("S2_kolm", loop_num, fields.S2_kolm) # save theoretical S2 for second order structure function
                            fields.save_Sp("S2_kolm_dir", loop_num, fields.S2_kolm_dir) # save theoretical S2 for second order structure function for direct cascade
                            fields.save_Sp("S2_from_spec", loop_num, fields.S2_from_spec) # save S2 calculated via hankel transform from spectrum

                        print(f'Calculating Sp done')


            # -- calculate signature function --
            # not all frames might be processed to save computing time (depends on which Sp_frames are processed)

            if "signature_function" in needed: # if signature function is needed

                if isinstance(pp.Sp_frames, str) and pp.Sp_frames == "all": # if all structure function frames were calculated

                    fields.find_V() # find signature function

                    if "signature_function" in requested: # if fluxes are requested
                        fields.save_signature_and_fluxes("V", loop_num, fields.V) # save signature function from S2
                        fields.save_signature_and_fluxes("V_hankel", loop_num, fields.V_hankel) # save signature function from Hankel transform of E(q)
                        fields.save_signature_and_fluxes("V_kolm", loop_num, fields.V_kolm) # save signature function from theoretical S2
                        fields.save_signature_and_fluxes("V_kolm_dir", loop_num, fields.V_kolm_dir) # save signature function from theoretical S2 for direct cascade

                else: # if only specific frames were processed

                    if loop_num in pp.Sp_frames: # if the loop number corresponds to a frame that should be processed

                        fields.find_V() # find signature function

                        if "signature_function" in requested: # if fluxes are requested
                            fields.save_signature_and_fluxes("V", loop_num, fields.V) # save signature function from S2
                            fields.save_signature_and_fluxes("V_hankel", loop_num, fields.V_hankel) # save signature function from Hankel transform of E(q)
                            fields.save_signature_and_fluxes("V_kolm", loop_num, fields.V_kolm) # save signature function from theoretical S2
                            fields.save_signature_and_fluxes("V_kolm_dir", loop_num, fields.V_kolm_dir) # save signature function from theoretical S2 for direct cascade



            # -- calculate fluxes --
            # not all frames might be processed to save computing time (depends on which Sp_frames are processed)

            if "fluxes" in needed: # if fluxes are needed

                if isinstance(pp.Sp_frames, str) and pp.Sp_frames == "all": # if all structure function frames were calculated

                    fields.find_fluxes() # find fluxes

                    if "fluxes" in requested: # if fluxes are requested
                        fields.save_signature_and_fluxes("Pi_V", loop_num, fields.Pi_V) # save real space flux
                        fields.save_signature_and_fluxes("Pi_E", loop_num, fields.Pi_E) # save spectral space flux
                        fields.save_signature_and_fluxes("T", loop_num, fields.T) # save spectral kinetic energy transfer function

                else: # if only specific frames were processed

                    if loop_num in pp.Sp_frames: # if the loop number corresponds to a frame that should be processed

                        fields.find_fluxes() # find fluxes

                        if "fluxes" in requested: # if fluxes are requested
                            fields.save_signature_and_fluxes("Pi_V", loop_num, fields.Pi_V) # save real space flux
                            fields.save_signature_and_fluxes("Pi_E", loop_num, fields.Pi_E) # save spectral space flux
                            fields.save_signature_and_fluxes("T", loop_num, fields.T) # save spectral kinetic energy transfer function


            # -- start new write --
            # set counter of writes one higher
            loop_num += 1

            end_pro_time = time.time()
            #print("Time for processing loop: ", end_pro_time - start_pro_time)


# ---- save evolutions to hdf5 file ----

# if any evolution data is requested, create a group "evolutions" for hdf5
if "total_quantities_evol" or "total_f_quantities_evol" or "system_scales" or "fields_evol" or "forcing_fields_evol" in requested:

    with h5py.File(fields.processed_path, "a") as f:

        # access or create group 'spectra'
        grp = f.require_group("evolutions")
        if "description" not in grp.attrs:
            grp.attrs["description"] = "all evolutions of the total quantities over time"

        # add time_evol
        if "time" in grp:
            del grp["time"]
        grp.create_dataset("time", data=time_evol)

        # add evolutions of total quantities
        if "total_quantities_evol" in requested:
            if "E_evol" in grp: # delete old evolutions
                del grp["E_evol"]
                del grp["F_evol"]
                del grp["Ediss_evol"]
                del grp["P_evol"]
                del grp["Fdiss_evol"]
            grp.create_dataset("E_evol", data=E_evol)
            grp.create_dataset("F_evol", data=F_evol)
            grp.create_dataset("Ediss_evol", data=Ediss_evol)
            grp.create_dataset("P_evol", data=P_evol)
            grp.create_dataset("Fdiss_evol", data=Fdiss_evol)
        
        # add evolutions of total forcing quantities
        if "total_f_quantities_evol" in requested:
            if "Ef_evol" in grp: # delete old evolutions
                del grp["Ef_evol"]
                del grp["Ff_evol"]
                del grp["Pf_evol"]
            grp.create_dataset("Ef_evol", data=Ef_evol)
            grp.create_dataset("Ff_evol", data=Ff_evol)
            grp.create_dataset("Pf_evol", data=Pf_evol)
            if "Ef_evol_theo" in grp: # delete old evolutions
                del grp["Ef_evol_theo"]
                del grp["Ff_evol_theo"]
                del grp["Pf_evol_theo"]
            grp.create_dataset("Ef_evol_theo", data=Ef_evol_theo)
            grp.create_dataset("Ff_evol_theo", data=Ff_evol_theo)
            grp.create_dataset("Pf_evol_theo", data=Pf_evol_theo)

        # add evolutions of system scales
        if "system_scales" in requested:
            if "q_Fdiss_evol" in grp: # delete old evolutions
                del grp["q_Fdiss_evol"]
                del grp["q_Fdiss_evol_theo"]
            grp.create_dataset("q_Fdiss_evol", data=q_Fdiss_evol)
            grp.create_dataset("q_Fdiss_evol_theo", data=q_Fdiss_evol_theo)
        
        # add standard deviations of real fields
        if "fields_evol" in requested:
            if "std_p_evol" in grp: # delete old evolutions
                del grp["std_p_evol"]
                del grp["std_w_evol"]
                del grp["std_s_evol"]
                del grp["std_ux_evol"]
                del grp["std_uy_evol"]
            grp.create_dataset("std_p_evol", data=std_p_evol)
            grp.create_dataset("std_w_evol", data=std_w_evol)
            grp.create_dataset("std_s_evol", data=std_s_evol)
            grp.create_dataset("std_ux_evol", data=std_ux_evol)
            grp.create_dataset("std_uy_evol", data=std_uy_evol)
       
        # add standard deviations of real forcing fields
        if "forcing_fields_evol" in requested:
            if "std_wf_evol" in grp: # delete old evolutions
                del grp["std_wf_evol"]
                del grp["std_sf_evol"]
                del grp["std_fx_evol"]
                del grp["std_fy_evol"]
            grp.create_dataset("std_wf_evol", data=std_wf_evol)
            grp.create_dataset("std_sf_evol", data=std_sf_evol)
            grp.create_dataset("std_fx_evol", data=std_fx_evol)
            grp.create_dataset("std_fy_evol", data=std_fy_evol)

stop_processing = time.time()

print(f'Processing ends at {datetime.now()}')
print(f'Total processing time: {stop_processing - start_processing}')