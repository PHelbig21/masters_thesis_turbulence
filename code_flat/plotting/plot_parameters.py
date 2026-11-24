'''
###################################################################################################################

PROCESSING FUNCTIONS

This file contains all parameters for plotting.
Goal: no need to change plot_functions, all parameters that need to be changed are listed in this file

Overview:
1. data access
2. real fields
3. spectral fields
4. spectra
5. structure function
6. evolutions

###################################################################################################################
'''



# __________ DATA ACCESS __________

run_dir = '/home/s-pahelb/Masterarbeit/Masterarbeit/flat_space/runs_used'
run_id = 1


# __________ WHAT TO PLOT __________

requested_plots = {
    "vorticity": True, 
    "vorticity_f": True,
    "pressure": True, 
    "stream": True, 
    "stream_f": True,
    "E_field": True, 
    "Ef_field": True,
    "spectra": True,
    "balance_spectra": True,
    "S3": False,
    "S2": False,
    "time_evolution": True,
    "balance_evolution": True,
    "flux_V": False,
    "flux_E": False,
    "signature_function": False
}


# __________ REAL FIELDS__________

# ---- vorticity ----
# w for vorticity
# wf for forcing vorticity

# -- figure size --
w_size = 4
wf_size = 4

# -- colour scaling --
# scaling of the colour bar (max, min of scale = +- w_scale_mult * w_std ) 
# w_std = mean standard deviation of vorticity over all frames
w_scale_mult = 2 # real number > 0
wf_scale_mult = 2 # real number > 0



# ---- pressure ----

# -- figure size --
p_size = 4

# -- colour scaling --
# scaling of the colour bar (max, min of scale = +- p_scale_mult * p_std ) 
# p_std = mean standard deviation of pressure over all frames
p_scale_mult = 2 # real number > 0



# ---- stream ----
# s for vorticity
# sf for forcing vorticity

# -- figure size --
s_size = 4
sf_size = 4

# -- colour scaling --
# scaling of the colour bar (max, min of scale = +- s_scale_mult * s_std ) 
# s_std = mean standard deviation of stream over all frames
s_scale_mult = 2 # real number > 0
sf_scale_mult = 2 # real number > 0



# __________ SPECTRAL FIELDS __________

# -- figure size --
E_field_size = 4
Ef_field_size = 4

# -- colour scaling --
# minimal and maximal value of the colorbar
E_field_bar_min = 1e-9
E_field_bar_max = 1e0
Ef_field_bar_min = 1e-7
Ef_field_bar_max = 1e2






# __________ SPECTRA __________

# ---- energy, enstrophy, palistrophy spectra ----

# -- figure size --
spectra_width = 12
spectra_height = 4

# -- y scaling --
# energy y-axis
E_spec_min = 3e-10
E_spec_max = 1e0
# enstrophy y-axis
F_spec_min = 3e-6
F_spec_max = 1e0
# palinstrophy y-axis
P_spec_min = 1e-5
P_spec_max = 1e0


# ---- dissipation and forcing spectra ----

# -- figure size --
diss_spectra_width = 8
diss_spectra_height = 4

# -- y scaling --
# energy y-axis
Ediss_spec_min = 3e-4*0.005
Ediss_spec_max = 1e2*0.005
# enstrophy y-axis
Fdiss_spec_min = 1e-1*0.005
Fdiss_spec_max = 1e4*0.005


# ---- energy transfer ----

# should the energy transfer function be included into balance spectra? (needs S3 to be calculated)
include_T = False





# __________ STRUCTURE FUNCTION __________

# -- figure size of S3 and S2 --
Sp_width = 5
Sp_height = 4

# -- y scaling --
# S3 y-axis
S3_min = -0.15
S3_max = 0.1
# S2 y-axis
S2_min = 0
S2_max = 1.0

# -- prefactor beta --
beta = 2.0 # prefactor for Kolmogorovs 2/3 law for S2





# __________ FLUXES AND SIGNATURE FUNCTION __________

# -- figure size of S3 and S2 --
flux_width = 5
flux_height = 4

V_width = 5
V_height = 4

Pi_E_min = -0.3
Pi_E_max = 0.3

Pi_V_min = -0.05
Pi_V_max = 0.05

V_min = -1
V_max = 1


# __________ EVOLUTIONS __________

# -- figure size --
evolutions_width = 12
evolutions_height = 4

bal_evolutions_width = 8
bal_evolutions_height = 4