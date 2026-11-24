'''
###################################################################################################################

PLOTTING

This file plots all post-processed simulation results from the hdf5 post-processed file.

Navigate to Folder: 
    cd flat_space/code

Execution:
    python3 plotting/plotting.py

Video creation:
go to e.g.
    cd ../runs/run_00020/results
then create videos
    ffmpeg -framerate 20 -i vorticity/w_%05d.png -c:v libx264 -pix_fmt yuv420p videos/vorticity.mp4
    ffmpeg -framerate 20 -i pressure/p_%05d.png -c:v libx264 -pix_fmt yuv420p videos/pressure.mp4
    ffmpeg -framerate 20 -i stream/s_%05d.png -c:v libx264 -pix_fmt yuv420p videos/stream.mp4
    ffmpeg -framerate 20 -i forcing_vorticity/wf_%05d.png -c:v libx264 -pix_fmt yuv420p videos/forcing_vorticity.mp4
    ffmpeg -framerate 20 -i forcing_stream/sf_%05d.png -c:v libx264 -pix_fmt yuv420p videos/forcing_stream.mp4
    ffmpeg -framerate 20 -i E_field/E_field_%05d.png -c:v libx264 -pix_fmt yuv420p videos/E_field.mp4
    ffmpeg -framerate 20 -i forcing_E_field/Ef_field_%05d.png -c:v libx264 -pix_fmt yuv420p videos/forcing_E_field.mp4
    ffmpeg -framerate 20 -i spectra/spectra_%05d.png -c:v libx264 -pix_fmt yuv420p videos/spectra.mp4
    ffmpeg -framerate 20 -i balance_spectra/spectrab_%05d.png -c:v libx264 -pix_fmt yuv420p videos/balance_spectra.mp4
    ffmpeg -framerate 2 -pattern_type glob -i 'structure_function_2/S2_*.png' \-c:v libx264 -pix_fmt yuv420p videos/structure_function_2.mp4
    ffmpeg -framerate 2 -pattern_type glob -i 'structure_function_3/S3_*.png' \-c:v libx264 -pix_fmt yuv420p videos/structure_function_3.mp4
    ffmpeg -framerate 2 -pattern_type glob -i 'signature_function/V_*.png' \-c:v libx264 -pix_fmt yuv420p videos/signature_function.mp4

Overview:
1. Imports
2. Plotting

###################################################################################################################
'''


# __________ IMPORTS __________

# external files
import plot_functions as pf
import plot_parameters as pp

import time # time measurement

# parallelisation
from joblib import Parallel, delayed

start_plotting = time.time()


# __________ PLOTTING __________


# list all frame IDs
frames = pf.find_frame_IDs(pp.run_id)


# plot time evolution
if pp.requested_plots["time_evolution"]:
    pf.plot_time_evolution(pp.run_id) # energy, enstrophy, palinstrophy
if pp.requested_plots["balance_evolution"]:
    pf.plot_balance_evolution(pp.run_id) # energy dissipation, enstrophy dissipation, forcing power, forcing enstrophy



def plot_frame(frame_id):

    # plot real fields
    if pp.requested_plots["vorticity"]:
        pf.plot_vorticity(pp.run_id, frame_id) # vorticity field
    if pp.requested_plots["vorticity_f"]:
        pf.plot_vorticity_f(pp.run_id, frame_id) # forcing vorticity field
    if pp.requested_plots["pressure"]:
        pf.plot_pressure(pp.run_id, frame_id) # pressure field
    if pp.requested_plots["stream"]:
        pf.plot_stream(pp.run_id, frame_id) # stream function
    if pp.requested_plots["stream_f"]:
        pf.plot_stream_f(pp.run_id, frame_id) # forcing stream function

    # plot spectral fields
    if pp.requested_plots["E_field"]:
        pf.plot_E_field(pp.run_id, frame_id) # energy field
    if pp.requested_plots["Ef_field"]:
        pf.plot_Ef_field(pp.run_id, frame_id) # forcing energy field

    # plot spectra
    if pp.requested_plots["spectra"]:
        pf.plot_spectra(pp.run_id, frame_id) # vorticity spectrum
    if pp.requested_plots["balance_spectra"]:
        pf.plot_balance_spectra(pp.run_id, frame_id) # forcing vorticity spectrum

    # plot structure function
    if pp.requested_plots["S3"]:
        pf.plot_S3(pp.run_id, frame_id)
    if pp.requested_plots["S2"]:
        pf.plot_S2(pp.run_id, frame_id)

    # plot signature function
    if pp.requested_plots["signature_function"]:
        pf.plot_signature_function(pp.run_id, frame_id)


    # plot fluxes
    if pp.requested_plots["flux_V"]:
        pf.plot_flux_V(pp.run_id, frame_id)
    if pp.requested_plots["flux_E"]:
        pf.plot_flux_E(pp.run_id, frame_id)


    print(f'Frame {frame_id}/{max(frames)} done.', flush=True)



# loop over all frames (parallelised)
Parallel(n_jobs=-1, backend="loky")(delayed(plot_frame)(frame_id) for frame_id in frames)

# finish
end_plotting = time.time()
print(f'Total plotting time: {end_plotting - start_plotting}')
    