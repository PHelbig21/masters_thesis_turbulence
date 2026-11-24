'''
###################################################################################################################

PROCESSING PARAMETERS

This file contains all parameters for the processing.

Overview:
1. data access
2. spectra
3. structure function

###################################################################################################################
'''



# __________ IMPORTS __________

import numpy as np



# __________ DATA ACCESS __________

#run_dir = '../runs_test' # folder in which run is saved
run_dir = '/home/s-pahelb/Masterarbeit/Masterarbeit/curved_space/runs'
run_id = 9 # number of run



# __________ WHAT TO PROCESS __________

# all raw data fields (w, p, wf) are automatically read and saved to hdf5 file

# dependencies (e.g. pro_spectra needs pro_E_field) are automatically considered in processing.py
# thus all booleans can be chosen independently
# if e.g. "spectra": True but "E_field": False, then "E_field" is calculated to provide data for "spectra", but not saved to hdf5 file

requested_data = {
    "fields": True,                     # s, u
    "forcing_fields": True,             # sf, f
    "E_field": True,                    # E_field
    "Ef_field": True,                   # Ef_field 
    "spectra": True,                    # E_spectrum, F_spectrum, Ediss_spectrum, P_spectrum, Fdiss_spectrum
    "f_spectra": True,                  # Ef_spectrum, Ff_spectrum, Pf_spectrum
    "total_quantities_evol": True,      # E_evol, F_evol, Ediss_evol, P_evol, Fdiss_evol
    "total_f_quantities_evol": True,    # Ef_evol, Ff_evol, Pf_evol, Ef_evol_theo, Ff_evol_theo, Pf_evol_theo
    "system_scales": True,              # q_Fdiss_evol, q_Fdiss_evol_theo
    "fields_evol": True,                # std_p_evol, std_w_evol, std_s_evol, std_ux_evol, std_uy_evol (important for real fields plots)
    "forcing_fields_evol": True,        # std_wf_evol, std_sf_evol, std_fx_evol, std_fy_evol (important for spectral fields plots)
    "Sp": True,                         # S3, S2, S3_kolm, S3_kolm_dir
    "fluxes": False,                     # Pi_V, Pi_E
    "signature_function": False         # V
}

# if "Sp" is True, define frames for which structure functions should be processed (might not be "all" to save computing time, as this calculation is very expensive)
# Sp_frames = "all" : all frames are processed
# Sp_frames = [2, 5, 17] : frames 2, 5, 17 are processed
# Sp_frames = np.arange(0, 100, 10) : frames 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 are processed
Sp_frames = np.arange(0, 500, 10)

# this is the fraction of how many pairs of points shall be considered for the structure functions
# chose a value between 0 and 1
# 0 means no pairs of points are considered
# 1 means all pairs of points are considered
# 0.1 means 10% of all pairs of points are considered
# used to save computing time but also makes the statistics less accurate (but pairs of points are chosen randomly)
# it does not help to set it too small, since the pair selection process still needs to consider every pair, so this is the bottleneck for very small pair percentages
# ulitmately one should try out which works best for every run
# for Nphi = 512 and Ntheta = 256: 0.001 gives 3.2 min per frame and 0.01 gives 4.4 min per frame on 18 cores
pair_percentage = 0.01 


# __________ SPECTRA __________

# ---- binning ----

# linear or logarithmic binning?
# True for logarithmic binning
# False for linear binning (is the natural choice, since we can simply bin by n)
ell_log_binning = True

# if logarithmic binning, then specify the number of bins per decade
ell_num_bins_per_decade = 15


# to improve binning accuracy (splits each wave number cell into (subdivisions x subdivisions) subcells)
# should be set such that q_num_bins << N/2 * q_subdivisions
# can be set to 1 if q_num_bins << N/2, but for q_num_bins ~ N/2 it should be set to a higher value to not get stupid binning boundary effects
ell_subdivisions = 20 #100 





# __________ STRUCTURE FUNCTION __________ # TODO

# ---- binning ----

# linear or logarithmic binning?
# True for logarithmic binning
# False for linear binning
l_log_binning = True

# if logarithmic binning, then specify the number of bins per decade
l_num_bins_per_decade = 15

# number of bins used for the structure function
l_num_bins = 100

# no subpixel grid in this case, since there are much more distinct distance values on the sphere than in flat space
# also it would be hard to implement with the current structure to calculate the Sp



# __________ SIGNATURE FUNCTION AND FLUXES __________ # TODO

# ---- smoothing ----

# decide whether raw S2 and S3 data should be smoothed before using them to calculate Pi_V, Pi_E and V (avoid noise in derivatives)
smooth = True
# if True, set filter length and order for savgol filter
# S3 and S2 are smoothed for the calculation of Pi_V, Pi_E and V with savgol filter to be able to calculate less noisy derivatives
filter_length = 8
filter_order = 3


# ---- derivative accuracy ----

# set accuracy for findiff derivatives
acc = 2