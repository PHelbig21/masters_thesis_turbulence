'''
###################################################################################################################

PLOTTING FUNCTIONS

This file contains all functions that plot the processed data.

Overview:
1. imports
2. real fields
3. spectral fields
4. spectra
5. structure function
6. evolutions

###################################################################################################################
'''



# __________ IMPORTS __________


# refresh imports (especially external files)
import importlib

# calculation
import numpy as np

# plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # colorbar position
from matplotlib.colors import LogNorm # logarithmic colorbar
from matplotlib.colors import Normalize

# file handling
import h5py
import os # folder management

# external files
import plot_parameters as pp
importlib.reload(pp)

# latex settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",           
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})




# __________ REAL FIELDS __________


# ---- vorticity ----

# plot a single vorticity frame
def plot_vorticity(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

        # make sure that data exists
        if "real_fields" not in file:
            raise KeyError("Group 'real_fields' not found in HDF5 file, needed for vorticity plot")
        if "w" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/w' not found, needed for vorticity plot")
        if "x_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/x_grid' not found, needed for vorticity plot")
        if "y_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/y_grid' not found, needed for vorticity plot")
        if "z_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/z_grid' not found, needed for vorticity plot")
        if "phi_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/x_grid' not found, needed for vorticity plot")
        if "theta_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/y_grid' not found, needed for vorticity plot")        
        if "evolutions" not in file:
            raise KeyError("Group 'evolutions' not found in HDF5 file, needed for vorticity plot")
        if "std_w_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/std_w_evol' not found, needed for vorticity plot")

        # read fields
        w = file['real_fields']['w'][f"{frame_num:05d}"] # read vorticity
        x = file['real_fields']['x_grid'] # read the real space x-grid (embedding)
        y = file['real_fields']['y_grid'] # read the real space y-grid (embedding)
        z = file['real_fields']['z_grid'] # read the real space z-grid (embedding)
        phi = file['real_fields']['phi_grid']
        theta = file['real_fields']['theta_grid']

        # read other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame
        w_std = np.mean(file['evolutions']['std_w_evol']) # standard deviation of vorticity over all frames to scale the colorbar


        fig = plt.figure(figsize=(pp.w_size, pp.w_size / 2), constrained_layout=False)

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], left=-0.08, right=1, bottom=-0.05, top=1.05, wspace=-0.08)

        ax1 = fig.add_subplot(gs[0, 0], projection="3d")
        ax2 = fig.add_subplot(gs[0, 1])

        norm = Normalize(vmin=-pp.w_scale_mult * w_std,vmax=pp.w_scale_mult * w_std)

        # 3D plot
        ax1.view_init(elev=40, azim=20)

        ax1.plot_surface(x, y, z, facecolors=plt.cm.turbo_r(norm(w)), rstride=1, cstride=1)

        ax1.set_axis_off()
        ax1.text2D(0.5, 0.85, 'vorticity', ha='center', fontsize=16, transform=ax1.transAxes)
        ax1.text2D(0.5, 0.15, f"$t$ = {time:.2f}", ha='center', fontsize=14, transform=ax1.transAxes)

        # 2D plot
        pcm = ax2.pcolormesh(phi, theta, w, cmap=plt.cm.turbo_r, vmin=-pp.w_scale_mult*w_std, vmax=pp.w_scale_mult*w_std, shading='auto')
        ax2.text(0.5, 1.15, 'vorticity', ha='center', fontsize=16, transform=ax2.transAxes)
        ax2.text(0.5, -0.2, f"$t$ = {time:.2f}", ha='center', fontsize=14, transform=ax2.transAxes)
        
        ax2.axis("off")
        ax2.set_aspect('equal', adjustable='box')

        # colorbar
        fig.colorbar(pcm, ax=ax2, shrink=0.5)
        
        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/vorticity/w_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)


# plot a single vorticity frame
def plot_vorticity_f(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

        # make sure that data exists
        if "real_fields" not in file:
            raise KeyError("Group 'real_fields' not found in HDF5 file, needed for vorticity plot")
        if "wf" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/wf' not found, needed for vorticity plot")
        if "x_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/x_grid' not found, needed for vorticity plot")
        if "y_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/y_grid' not found, needed for vorticity plot")
        if "z_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/z_grid' not found, needed for vorticity plot")
        if "phi_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/x_grid' not found, needed for vorticity plot")
        if "theta_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/y_grid' not found, needed for vorticity plot")        
        if "evolutions" not in file:
            raise KeyError("Group 'evolutions' not found in HDF5 file, needed for vorticity plot")
        if "std_wf_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/std_wf_evol' not found, needed for vorticity plot")

        # read fields
        wf = file['real_fields']['wf'][f"{frame_num:05d}"] # read vorticity
        x = file['real_fields']['x_grid'] # read the real space x-grid (embedding)
        y = file['real_fields']['y_grid'] # read the real space y-grid (embedding)
        z = file['real_fields']['z_grid'] # read the real space z-grid (embedding)
        phi = file['real_fields']['phi_grid']
        theta = file['real_fields']['theta_grid']

        # read other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame
        wf_std = np.mean(file['evolutions']['std_wf_evol']) # standard deviation of vorticity over all frames to scale the colorbar


        fig = plt.figure(figsize=(pp.wf_size, pp.wf_size / 2), constrained_layout=False)

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], left=-0.08, right=1, bottom=-0.05, top=1.05, wspace=-0.08)

        ax1 = fig.add_subplot(gs[0, 0], projection="3d")
        ax2 = fig.add_subplot(gs[0, 1])

        norm = Normalize(vmin=-pp.wf_scale_mult * wf_std,vmax=pp.wf_scale_mult * wf_std)

        # 3D plot
        ax1.view_init(elev=40, azim=20)

        ax1.plot_surface(x, y, z, facecolors=plt.cm.turbo_r(norm(wf)), rstride=1, cstride=1)

        ax1.set_axis_off()
        ax1.text2D(0.5, 0.85, 'forcing vorticity', ha='center', fontsize=16, transform=ax1.transAxes)
        ax1.text2D(0.5, 0.15, f"$t$ = {time:.2f}", ha='center', fontsize=14, transform=ax1.transAxes)

        # 2D plot
        pcm = ax2.pcolormesh(phi, theta, wf, cmap=plt.cm.turbo_r, vmin=-pp.wf_scale_mult*wf_std, vmax=pp.wf_scale_mult*wf_std, shading='auto')
        ax2.text(0.5, 1.15, 'forcing vorticity', ha='center', fontsize=16, transform=ax2.transAxes)
        ax2.text(0.5, -0.2, f"$t$ = {time:.2f}", ha='center', fontsize=14, transform=ax2.transAxes)
        
        ax2.axis("off")
        ax2.set_aspect('equal', adjustable='box')

        # colorbar
        fig.colorbar(pcm, ax=ax2, shrink=0.5)
        
        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/forcing_vorticity/wf_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)



# ---- pressure ----

# plot a single pressure frame
def plot_pressure(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

        # make sure that data exists
        if "real_fields" not in file:
            raise KeyError("Group 'real_fields' not found in HDF5 file, needed for vorticity plot")
        if "p" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/p' not found, needed for vorticity plot")
        if "x_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/x_grid' not found, needed for vorticity plot")
        if "y_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/y_grid' not found, needed for vorticity plot")
        if "z_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/z_grid' not found, needed for vorticity plot")
        if "phi_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/x_grid' not found, needed for vorticity plot")
        if "theta_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/y_grid' not found, needed for vorticity plot")        
        if "evolutions" not in file:
            raise KeyError("Group 'evolutions' not found in HDF5 file, needed for vorticity plot")
        if "std_w_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/std_w_evol' not found, needed for vorticity plot")

        # read fields
        p = file['real_fields']['p'][f"{frame_num:05d}"] # read vorticity
        x = file['real_fields']['x_grid'] # read the real space x-grid (embedding)
        y = file['real_fields']['y_grid'] # read the real space y-grid (embedding)
        z = file['real_fields']['z_grid'] # read the real space z-grid (embedding)
        phi = file['real_fields']['phi_grid']
        theta = file['real_fields']['theta_grid']

        # read other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame
        p_std = np.mean(file['evolutions']['std_p_evol']) # standard deviation of vorticity over all frames to scale the colorbar


        fig = plt.figure(figsize=(pp.p_size, pp.p_size / 2), constrained_layout=False)

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], left=-0.08, right=1, bottom=-0.05, top=1.05, wspace=-0.08)

        ax1 = fig.add_subplot(gs[0, 0], projection="3d")
        ax2 = fig.add_subplot(gs[0, 1])

        norm = Normalize(vmin=-pp.p_scale_mult * p_std,vmax=pp.p_scale_mult * p_std)

        # 3D plot
        ax1.view_init(elev=40, azim=20)

        ax1.plot_surface(x, y, z, facecolors=plt.cm.turbo_r(norm(p)), rstride=1, cstride=1)

        ax1.set_axis_off()
        ax1.text2D(0.5, 0.85, 'pressure', ha='center', fontsize=16, transform=ax1.transAxes)
        ax1.text2D(0.5, 0.15, f"$t$ = {time:.2f}", ha='center', fontsize=14, transform=ax1.transAxes)

        # 2D plot
        pcm = ax2.pcolormesh(phi, theta, p, cmap=plt.cm.turbo_r, vmin=-pp.p_scale_mult*p_std, vmax=pp.p_scale_mult*p_std, shading='auto')
        ax2.text(0.5, 1.15, 'pressure', ha='center', fontsize=16, transform=ax2.transAxes)
        ax2.text(0.5, -0.2, f"$t$ = {time:.2f}", ha='center', fontsize=14, transform=ax2.transAxes)
        
        ax2.axis("off")
        ax2.set_aspect('equal', adjustable='box')

        # colorbar
        fig.colorbar(pcm, ax=ax2, shrink=0.5)
        
        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/pressure/p_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)



# ---- stream function ----

# plot a single stream frame
def plot_stream(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

                # make sure that data exists
        if "real_fields" not in file:
            raise KeyError("Group 'real_fields' not found in HDF5 file, needed for vorticity plot")
        if "s" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/s' not found, needed for vorticity plot")
        if "x_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/x_grid' not found, needed for vorticity plot")
        if "y_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/y_grid' not found, needed for vorticity plot")
        if "z_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/z_grid' not found, needed for vorticity plot")
        if "phi_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/x_grid' not found, needed for vorticity plot")
        if "theta_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/y_grid' not found, needed for vorticity plot")        
        if "evolutions" not in file:
            raise KeyError("Group 'evolutions' not found in HDF5 file, needed for vorticity plot")
        if "std_s_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/std_s_evol' not found, needed for vorticity plot")

        # read fields
        s = file['real_fields']['s'][f"{frame_num:05d}"] # read vorticity
        x = file['real_fields']['x_grid'] # read the real space x-grid (embedding)
        y = file['real_fields']['y_grid'] # read the real space y-grid (embedding)
        z = file['real_fields']['z_grid'] # read the real space z-grid (embedding)
        phi = file['real_fields']['phi_grid']
        theta = file['real_fields']['theta_grid']

        # read other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame
        s_std = np.mean(file['evolutions']['std_s_evol']) # standard deviation of vorticity over all frames to scale the colorbar


        fig = plt.figure(figsize=(pp.s_size, pp.s_size / 2), constrained_layout=False)

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], left=-0.08, right=1, bottom=-0.05, top=1.05, wspace=-0.08)

        ax1 = fig.add_subplot(gs[0, 0], projection="3d")
        ax2 = fig.add_subplot(gs[0, 1])

        norm = Normalize(vmin=-pp.s_scale_mult * s_std,vmax=pp.s_scale_mult * s_std)

        # 3D plot
        ax1.view_init(elev=40, azim=20)

        ax1.plot_surface(x, y, z, facecolors=plt.cm.turbo_r(norm(s)), rstride=1, cstride=1)

        ax1.set_axis_off()
        ax1.text2D(0.5, 0.85, 'stream', ha='center', fontsize=16, transform=ax1.transAxes)
        ax1.text2D(0.5, 0.15, f"$t$ = {time:.2f}", ha='center', fontsize=14, transform=ax1.transAxes)

        # 2D plot
        pcm = ax2.pcolormesh(phi, theta, s, cmap=plt.cm.turbo_r, vmin=-pp.s_scale_mult*s_std, vmax=pp.s_scale_mult*s_std, shading='auto')
        ax2.text(0.5, 1.15, 'stream', ha='center', fontsize=16, transform=ax2.transAxes)
        ax2.text(0.5, -0.2, f"$t$ = {time:.2f}", ha='center', fontsize=14, transform=ax2.transAxes)
        
        ax2.axis("off")
        ax2.set_aspect('equal', adjustable='box')

        # colorbar
        fig.colorbar(pcm, ax=ax2, shrink=0.5)
        
        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/stream/s_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)



# plot a single forcing stream frame
def plot_stream_f(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

       # -- data --

        # make sure that data exists
        if "real_fields" not in file:
            raise KeyError("Group 'real_fields' not found in HDF5 file, needed for vorticity plot")
        if "sf" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/sf' not found, needed for vorticity plot")
        if "x_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/x_grid' not found, needed for vorticity plot")
        if "y_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/y_grid' not found, needed for vorticity plot")
        if "z_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/z_grid' not found, needed for vorticity plot")
        if "phi_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/x_grid' not found, needed for vorticity plot")
        if "theta_grid" not in file["real_fields"]:
            raise KeyError("Dataset 'real_fields/y_grid' not found, needed for vorticity plot")        
        if "evolutions" not in file:
            raise KeyError("Group 'evolutions' not found in HDF5 file, needed for vorticity plot")
        if "std_sf_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/std_sf_evol' not found, needed for vorticity plot")

        # read fields
        sf = file['real_fields']['sf'][f"{frame_num:05d}"] # read vorticity
        x = file['real_fields']['x_grid'] # read the real space x-grid (embedding)
        y = file['real_fields']['y_grid'] # read the real space y-grid (embedding)
        z = file['real_fields']['z_grid'] # read the real space z-grid (embedding)
        phi = file['real_fields']['phi_grid']
        theta = file['real_fields']['theta_grid']

        # read other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame
        sf_std = np.mean(file['evolutions']['std_sf_evol']) # standard deviation of vorticity over all frames to scale the colorbar


        fig = plt.figure(figsize=(pp.sf_size, pp.sf_size / 2), constrained_layout=False)

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], left=-0.08, right=1, bottom=-0.05, top=1.05, wspace=-0.08)

        ax1 = fig.add_subplot(gs[0, 0], projection="3d")
        ax2 = fig.add_subplot(gs[0, 1])

        norm = Normalize(vmin=-pp.sf_scale_mult * sf_std,vmax=pp.sf_scale_mult * sf_std)

        # 3D plot
        ax1.view_init(elev=40, azim=20)

        ax1.plot_surface(x, y, z, facecolors=plt.cm.turbo_r(norm(sf)), rstride=1, cstride=1)

        ax1.set_axis_off()
        ax1.text2D(0.5, 0.85, 'forcing stream', ha='center', fontsize=16, transform=ax1.transAxes)
        ax1.text2D(0.5, 0.15, f"$t$ = {time:.2f}", ha='center', fontsize=14, transform=ax1.transAxes)

        # 2D plot
        pcm = ax2.pcolormesh(phi, theta, sf, cmap=plt.cm.turbo_r, vmin=-pp.sf_scale_mult*sf_std, vmax=pp.sf_scale_mult*sf_std, shading='auto')
        ax2.text(0.5, 1.15, 'forcing stream', ha='center', fontsize=16, transform=ax2.transAxes)
        ax2.text(0.5, -0.2, f"$t$ = {time:.2f}", ha='center', fontsize=14, transform=ax2.transAxes)
        
        ax2.axis("off")
        ax2.set_aspect('equal', adjustable='box')

        # colorbar
        fig.colorbar(pcm, ax=ax2, shrink=0.5)
        
        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/forcing_stream/sf_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)




# __________ SPECTRAL FIELDS __________


# ---- spectral kinetic energy field ----

# plot a single kinetic energy frame
def plot_E_field(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

        if "spectral_fields" not in file:
            raise KeyError("Group 'spectral_fields' not found in HDF5 file, needed for E_field plot")
        if "E_field" not in file["spectral_fields"]:
            raise KeyError("Dataset 'spectral_fields/E_field' not found, needed for E_field plot")
        if "ell_grid" not in file["spectral_fields"]:
            raise KeyError("Dataset 'spectral_fields/ell_grid' not found, needed for E_field plot")
        if "m_grid" not in file["spectral_fields"]:
            raise KeyError("Dataset 'spectral_fields/m_grid' not found, needed for E_field plot")

        # read fields
        E_field = file['spectral_fields']['E_field'][f"{frame_num:05d}"] # read E_field
        ell = file['spectral_fields']['ell_grid'] # read the real space x-grid
        m = file['spectral_fields']['m_grid'] # read the real space y-grid

        # read other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame


        # -- preparation --
        
        # make np arrays to be able to apply numpy indexing
        ell = np.asarray(ell)
        m = np.asarray(m)
        E_field = np.asarray(E_field)
        # order Ef_field into Ef_ordered to use imshow
        ell_max = ell.max()
        m_max = m.max() 
        E_ordered = np.full((ell_max + 1, m_max + 1), np.nan) 
        E_ordered[ell, m] = E_field   


        # -- plotting --

        # figure size
        fig, ax = plt.subplots(figsize=(pp.E_field_size, pp.E_field_size))
        ax.set_aspect('equal') # equal aspect ratio

        # plot data
        im = ax.imshow(E_ordered, origin='lower', aspect='auto', cmap='hot', norm = LogNorm(vmin=pp.E_field_bar_min, vmax=pp.E_field_bar_max))

        # colourbar
        #  axis fits to plot height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        # label
        cbar.set_label(r'$E_{n,m}$')

        # axis labels
        ax.set_xlabel(r'order $m$')
        ax.set_ylabel(r'degree $n$')

        # information
        fig.text(0.5, 0.91, 'spectral kinetic energy field', ha='center', fontsize=14) # title
        fig.text(0.25, 0.8, rf"$t$ = {time:.2f}", ha='center', fontsize=12, color='white') # time information

        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/E_field/E_field_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)



# plot a single kinetic energy frame
def plot_Ef_field(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

        if "spectral_fields" not in file:
            raise KeyError("Group 'spectral_fields' not found in HDF5 file, needed for Ef_field plot")
        if "Ef_field" not in file["spectral_fields"]:
            raise KeyError("Dataset 'spectral_fields/Ef_field' not found, needed for Ef_field plot")
        if "ell_grid" not in file["spectral_fields"]:
            raise KeyError("Dataset 'spectral_fields/ell_grid' not found, needed for Ef_field plot")
        if "m_grid" not in file["spectral_fields"]:
            raise KeyError("Dataset 'spectral_fields/m_grid' not found, needed for Ef_field plot")

        # read fields
        Ef_field = file['spectral_fields']['Ef_field'][f"{frame_num:05d}"] # read E_field
        ell = file['spectral_fields']['ell_grid'] # read the real space x-grid
        m = file['spectral_fields']['m_grid'] # read the real space y-grid

        # read other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame


        # -- preparation --

        # make np arrays to be able to apply numpy indexing
        ell = np.asarray(ell)
        m = np.asarray(m)
        Ef_field = np.asarray(Ef_field)
        # order Ef_field into Ef_ordered to use imshow
        ell_max = ell.max()
        m_max = m.max()
        Ef_ordered = np.full((ell_max + 1, m_max + 1), np.nan)
        Ef_ordered[ell, m] = Ef_field        


        # -- plotting --

        # figure size
        fig, ax = plt.subplots(figsize=(pp.Ef_field_size, pp.Ef_field_size))
        ax.set_aspect('equal') # equal aspect ratio

        # plot data
        im = ax.imshow(Ef_ordered, origin='lower', aspect='auto', cmap='hot', norm = LogNorm(vmin=pp.Ef_field_bar_min, vmax=pp.Ef_field_bar_max))

        # colourbar
        #  axis fits to plot height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        # label
        cbar.set_label(r'$E^f_{n,m}$')

        # axis labels
        ax.set_xlabel(r'order $m$')
        ax.set_ylabel(r'degree $n$')

        # information
        fig.text(0.5, 0.91, 'spectral kinetic forcing power field', ha='center', fontsize=14) # title
        fig.text(0.25, 0.8, rf"$t$ = {time:.2f}", ha='center', fontsize=12, color='white') # time information

        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/forcing_E_field/Ef_field_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)




# __________ SPECTRA __________

# ---- energy, enstrophy, and palinstrophy spectra ----

# plot a single spectrum frame
def plot_spectra(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

        # check if data exists
        if "spectra" not in file:
            raise KeyError("Group 'spectra' not found in HDF5 file, needed for spectra plot")
        if "E_spectrum" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/E_spectrum' not found, needed for spectra plot")
        if "F_spectrum" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/F_spectrum' not found, needed for spectra plot")
        if "P_spectrum" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/P_spectrum' not found, needed for spectra plot")
        if "E_spectrum_init" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/E_spectrum_init' not found, needed for spectra plot")
        if "F_spectrum_init" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/F_spectrum_init' not found, needed for spectra plot")
        if "P_spectrum_init" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/P_spectrum_init' not found, needed for spectra plot")
        if "modes" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/modes' not found, needed for spectra plot")
        if "evolutions" not in file:
            raise KeyError("Group 'evolutions' not found in HDF5 file, needed for spectra plot")
        if "ell_Fdiss_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/ell_Fdiss_evol' not found, needed for spectra plot")
        if "ell_Fdiss_evol_theo" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/ell_Fdiss_evol_theo' not found, needed for spectra plot")

        # read spectra
        E_spectrum = file['spectra']['E_spectrum'][f"{frame_num:05d}"][()] # read E_spectrum (energy)
        F_spectrum = file['spectra']['F_spectrum'][f"{frame_num:05d}"][()] # read F_spectrum (enstrophy)
        P_spectrum = file['spectra']['P_spectrum'][f"{frame_num:05d}"][()] # read P_spectrum (palinstrophy)

        # read initial theoretical spectra
        E_spectrum_init = file['spectra']['E_spectrum_init'][()] # read E_spectrum_init (initial energy)
        F_spectrum_init = file['spectra']['F_spectrum_init'][()] # read F_spectrum_init (initial enstrophy)
        P_spectrum_init = file['spectra']['P_spectrum_init'][()] # read P_spectrum_init (initial palinstrophy)

        # read absolute wavenumber
        ell = file['spectra']['modes'][()]

        # read other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame
        max_ell = np.max(ell)
        ell_Fdiss = file['evolutions']['ell_Fdiss_evol'][file['evolutions']['time'] == time]
        ell_Fdiss_theo = file['evolutions']['ell_Fdiss_evol_theo'][file['evolutions']['time'] == time]
        ellf = file['parameters']['forcing']['ellf'][()]

        # reference slope for ideal turbulence
        slopeE = ell**(-5/3)
        slopeF = ell**(1/3)
        slopeP = ell**(7/3)
        slopeE_dir = ell**(-3)
        slopeF_dir = ell**(-1)
        slopeP_dir = ell**(1)

        # -- plotting --

        # plot with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(pp.spectra_width, pp.spectra_height))

        # plot energy spectrum
        # grid
        ax1.grid(alpha=0.5)
        # data
        ax1.plot(ell, E_spectrum, color='tab:blue', label = r'$\tilde{E}_n$')
        ax1.plot(ell, E_spectrum_init, color='tab:blue', label = r'$E^0_{n}$', ls='dotted', alpha = 0.5)
        # reference slope
        ax1.plot(ell, slopeE, ls='--', color='grey', alpha=0.7)
        ax1.plot(ell, slopeE_dir, ls='--', color='grey', alpha=0.7)
        ax1.text(ell[-len(ell)//2], slopeE[-len(slopeE)//2]*1.5, r'$~n^{-5/3}$', fontsize=12, color='grey', ha='left', va='bottom')
        ax1.text(ell[-len(ell)//2], slopeE_dir[-len(slopeE_dir)//2]*1.5, r'$~n^{-3}$', fontsize=12, color='grey', ha='left', va='bottom')
        # enstrophy dissipation scale
        ax1.axvline(x=ell_Fdiss_theo, color='grey', alpha=0.2, ls='dotted')
        ax1.text(ell_Fdiss_theo*1.05, pp.E_spec_min, r'$q_\eta$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        ax1.axvline(x=ell_Fdiss, color='grey', alpha=0.2)
        ax1.text(ell_Fdiss*1.05, pp.E_spec_min, r'$\tilde{q}_\eta$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # forcing scale
        ax1.axvline(x=ellf, color='grey', alpha=0.2, ls='dotted')
        ax1.text(ellf*1.05, pp.E_spec_min, r'$q_f$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # title
        ax1.set_title('kinetic energy', fontsize=14)
        # axis
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(1, max_ell)
        ax1.set_ylim(pp.E_spec_min, pp.E_spec_max)
        # lables
        ax1.set_xlabel(r'degree $n$', fontsize=12)
        ax1.legend(loc='lower left')

        # plot enstrophy spectrum
        # grid
        ax2.grid(alpha=0.5)
        # data
        ax2.plot(ell, F_spectrum, color='tab:red', label = r'$\tilde{\Omega}_n$')
        ax2.plot(ell, F_spectrum_init, color='tab:red', label = r'$\Omega^0_{n}$', ls='dotted', alpha = 0.5)
        # reference slope
        ax2.plot(ell, slopeF, ls='--', color='grey', alpha=0.7)
        ax2.plot(ell, slopeF_dir, ls='--', color='grey', alpha=0.7)
        ax2.text(ell[-len(ell)//2], slopeF[-len(slopeF)//2]*1.3, r'$~n^{1/3}$', fontsize=12, color='grey', ha='left', va='bottom')
        ax2.text(ell[-len(ell)//2], slopeF_dir[-len(slopeF_dir)//2]*1.3, r'$~n^{-1}$', fontsize=12, color='grey', ha='left', va='bottom')
        # enstrophy dissipation scale
        ax2.axvline(x=ell_Fdiss_theo, color='grey', alpha=0.2, ls='dotted')
        ax2.text(ell_Fdiss_theo*1.05, pp.F_spec_min, r'$n_\eta$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        ax2.axvline(x=ell_Fdiss, color='grey', alpha=0.2)
        ax2.text(ell_Fdiss*1.05, pp.F_spec_min, r'$\tilde{n}_\eta$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # forcing scale
        ax2.axvline(x=ellf, color='grey', alpha=0.2, ls='dotted')
        ax2.text(ellf*1.05, pp.F_spec_min, r'$n_f$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # title
        ax2.set_title('enstrophy', fontsize=14)
        # axis
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(1, max_ell)
        ax2.set_ylim(pp.F_spec_min, pp.F_spec_max)
        # lables
        ax2.set_xlabel(r'degree $n$', fontsize=12)
        ax2.legend(loc='lower left')

        # plot palistrophy spectrum
        # grid
        ax3.grid(alpha=0.5)
        # data
        ax3.plot(ell, P_spectrum, color='tab:green', label = r'$\tilde{P}_n$')
        ax3.plot(ell, P_spectrum_init, color='tab:green', label = r'$P^0_{n}$', ls='dotted', alpha = 0.5)
        # reference slope
        ax3.plot(ell, slopeP, ls='--', color='grey', alpha=0.7)
        ax3.plot(ell, slopeP_dir, ls='--', color='grey', alpha=0.7)
        ax3.text(ell[-len(ell)//2], slopeP[-len(slopeP)//2], r'$~n^{7/3}$', fontsize=12, color='grey', ha='right', va='bottom')
        ax3.text(ell[-len(ell)//2], slopeP_dir[-len(slopeP_dir)//2], r'$~n^{1}$', fontsize=12, color='grey', ha='right', va='bottom')
        # enstrophy dissipation scale
        ax3.axvline(x=ell_Fdiss_theo, color='grey', alpha=0.2, ls='dotted')
        ax3.text(ell_Fdiss_theo*1.05, pp.P_spec_min, r'$n_\eta$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        ax3.axvline(x=ell_Fdiss, color='grey', alpha=0.2)
        ax3.text(ell_Fdiss*1.05, pp.P_spec_min, r'$\tilde{n}_\eta$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # forcing scale
        ax3.axvline(x=ellf, color='grey', alpha=0.2, ls='dotted')
        ax3.text(ellf*1.05, pp.P_spec_min, r'$n_f$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # title
        ax3.set_title('palinstrophy', fontsize=14)
        # axis
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlim(1, max_ell)
        ax3.set_ylim(pp.P_spec_min, pp.P_spec_max)
        # lables
        ax3.set_xlabel(r'degree $n$', fontsize=12)
        ax3.legend(loc='lower left')

        # time
        fig.subplots_adjust(bottom=0.25)
        fig.text(0.5, 0.05, f"time: $t$ = {time:.2f}", ha='center', fontsize=12)

        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/spectra/spectra_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)



# ---- dissipation and forcing spectra ----

# plot a single spectrum frame
def plot_balance_spectra(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

        if "spectra" not in file:
            raise KeyError("Group 'spectra' not found in HDF5 file, needed for balance_spectra plot")
        if "Ediss_spectrum" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/Ediss_spectrum' not found, needed for balance_spectra plot")
        if "Fdiss_spectrum" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/Fdiss_spectrum' not found, needed for balance_spectra plot")
        if "Ef_spectrum" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/Ef_spectrum' not found, needed for balance_spectra plot")
        if "Ff_spectrum" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/Ff_spectrum' not found, needed for balance_spectra plot")
        if "Ef_spectrum_theo" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/Ef_spectrum_theo' not found, needed for balance_spectra plot")
        if "Ff_spectrum_theo" not in file["spectra"]:
            raise KeyError("Dataset 'spectra/Ff_spectrum_theo' not found, needed for balance_spectra plot")
        if "modes" not in file["spectra"]:
            raise KeyError("Dataset 'modes/wavenumber' not found, needed for balance_spectra plot")      
        if "evolutions" not in file:
            raise KeyError("Group 'evolutions' not found in HDF5 file, needed for balance_spectra plot")
        if "ell_Fdiss_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/ell_Fdiss_evol' not found, needed for balance_spectra plot")
        if "ell_Fdiss_evol_theo" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/ell_Fdiss_evol_theo' not found, needed for balance_spectra plot")
        if pp.include_T:
            if "signature_and_fluxes" not in file:
                raise KeyError("Group 'signature_and_fluxes' not found in HDF5 file, needed for balance_spectra plot if include_T is True")
            if "T" not in file['signature_and_fluxes']:
                raise KeyError("Dataset 'signature_and_fluxes/T' not found, needed for balance_spectra plot if include_T is True")

        # read spectra
        Ediss_spectrum = file['spectra']['Ediss_spectrum'][f"{frame_num:05d}"][()] # read Ediss_spectrum (energy dissipation)
        Fdiss_spectrum = file['spectra']['Fdiss_spectrum'][f"{frame_num:05d}"][()] # read Fdiss_spectrum (enstrophy dissipation)
        Ef_spectrum = file['spectra']['Ef_spectrum'][f"{frame_num:05d}"][()] # read Ef_spectrum (kinetic forcing power)
        Ff_spectrum = file['spectra']['Ff_spectrum'][f"{frame_num:05d}"][()] # read Ff_spectrum (enstrophy forcing power)
        if pp.include_T:
            T_spectrum = file['signature_and_fluxes']['T'][f"{frame_num:05d}"][()]
            qT = file['signature_and_fluxes']['modes'][()]

        # read theoretical spectra
        Ef_spectrum_theo = file['spectra']['Ef_spectrum_theo'][()] # read Ef_spectrum_theo (theoretical kinetic forcing power)
        Ff_spectrum_theo = file['spectra']['Ff_spectrum_theo'][()] # read Ff_spectrum_theo (theoretical forcing enstrophy)

        # read absolute wavenumber
        ell = file['spectra']['modes'][()]

        # read other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame
        max_ell = np.max(ell)
        ell_Fdiss = file['evolutions']['ell_Fdiss_evol'][file['evolutions']['time'] == time]
        ell_Fdiss_theo = file['evolutions']['ell_Fdiss_evol_theo'][file['evolutions']['time'] == time]
        ellf = file['parameters']['forcing']['ellf'][()]

        # -- plotting --

        # plot with three subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(pp.diss_spectra_width, pp.diss_spectra_height))

        # plot energy spectrum
        # grid
        ax1.grid(alpha=0.5)
        # data
        ax1.plot(ell, Ediss_spectrum, color='tab:red', label = r'$2\kappa\tilde{\Omega}_n$')
        ax1.plot(ell, Ef_spectrum, color='tab:red', label = r'$\tilde{E}^f_n$', alpha = 0.5)
        ax1.plot(ell, Ef_spectrum_theo, color='tab:red', label = r'$E^f_n$', alpha = 0.5, ls='dotted')
        if pp.include_T:
            ax1.plot(qT, T_spectrum, color='tab:red', label = r'$\tilde{T}_n$', alpha = 0.7)
        # enstrophy dissipation scale
        ax1.axvline(x=ell_Fdiss_theo, color='grey', alpha=0.2, ls='dotted')
        ax1.text(ell_Fdiss_theo*1.05, pp.Ediss_spec_min, r'$n_\eta$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        ax1.axvline(x=ell_Fdiss, color='grey', alpha=0.2)
        ax1.text(ell_Fdiss*1.05, pp.Ediss_spec_min, r'$\tilde{n}_\eta$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # forcing scale
        ax1.axvline(x=ellf, color='grey', alpha=0.2, ls='dotted')
        ax1.text(ellf*1.05, pp.Ediss_spec_min, r'$n_f$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # title
        ax1.set_title('kinetic energy balance', fontsize=14)
        # axis
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(1, max_ell)
        ax1.set_ylim(pp.Ediss_spec_min, pp.Ediss_spec_max)
        # lables
        ax1.set_xlabel(r'degree $n$', fontsize=12)
        ax1.legend(loc='lower left')

        # plot enstrophy spectrum
        # grid
        ax2.grid(alpha=0.5)
        # data
        ax2.plot(ell, Fdiss_spectrum, color='tab:green', label = r'$2\kappa\tilde{P}_n$')
        ax2.plot(ell, Ff_spectrum, color='tab:green', label = r'$\tilde{\Omega}^f_n$', alpha = 0.5)
        ax2.plot(ell, Ff_spectrum_theo, color='tab:green', label = r'$\Omega^f_n$', alpha = 0.5, ls='dotted')
        # enstrophy dissipation scale
        ax2.axvline(x=ell_Fdiss_theo, color='grey', alpha=0.2, ls = 'dotted')
        ax2.text(ell_Fdiss_theo*1.05, pp.Fdiss_spec_min, r'$n_\eta$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        ax2.axvline(x=ell_Fdiss, color='grey', alpha=0.2)
        ax2.text(ell_Fdiss*1.05, pp.Fdiss_spec_min, r'$\tilde{n}_\eta$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # forcing scale
        ax2.axvline(x=ellf, color='grey', alpha=0.2, ls='dotted')
        ax2.text(ellf*1.05, pp.Fdiss_spec_min, r'$n_f$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # title
        ax2.set_title('enstrophy balance', fontsize=14)
        # axis
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(1, max_ell)
        ax2.set_ylim(pp.Fdiss_spec_min, pp.Fdiss_spec_max)
        # lables
        ax2.set_xlabel(r'degree $n$', fontsize=12)
        ax2.legend(loc='lower left')

        # time
        fig.subplots_adjust(bottom=0.25)
        fig.text(0.5, 0.05, f"time: $t$ = {time:.2f}", ha='center', fontsize=12)

        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/balance_spectra/spectrab_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)




# __________ STRUCTURE FUNCTION __________


# ---- structure function and Kolmogorovs law ----

# plot a single S3 frame
def plot_S3(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        #print('Plotting S3')

        # -- data --

        # read S3
        try: # try to read data
            S3 = file['structure_function']['S3'][f"{frame_num:05d}"][()] # read S3
            S3_kolm_flat = file['structure_function']['S3_kolm_flat'][f"{frame_num:05d}"][()] # read Kolmogorovs predicted S3
            S3_kolm_dir_flat = file['structure_function']['S3_kolm_dir_flat'][f"{frame_num:05d}"][()] # read predicted S3 for small correlation lengths
            S3_kolm_curved = file['structure_function']['S3_kolm_curved'][f"{frame_num:05d}"][()] # read S3 calculated from predicition in Bachelors thesis
        except KeyError: # if data does not exist (not all frames are processed) -> skip this function
            #print('S3 not available')
            return

        # read length
        l = file['structure_function']['length'][()]

        # energy and enstrophy dissipation
        E_diss = file['evolutions']['Ediss_evol'][frame_num]
        F_diss = file['evolutions']['Fdiss_evol'][frame_num]

        # other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame
        ellf = file['parameters']['forcing']['ellf'][()]


        # -- plotting --

        # size
        fig, ax = plt.subplots(1, 2, figsize=(2*pp.Sp_width, pp.Sp_height), constrained_layout=True)

        # original representation
        # data
        ax[0].plot(l[:-1], S3[:-1], label=r'$S^{3}_{\parallel}$', color="tab:blue") # structure function from velocity field
        ax[0].plot(l[:-1], S3_kolm_flat[:-1], label=r'$-\frac{3}{2}\varepsilon l$', color="tab:blue", alpha=0.5) # structure function from energy dissipation (Kolmogorovs law)
        ax[0].plot(l[:-1], S3_kolm_dir_flat[:-1], label=r'$-\frac{1}{8}\eta l^3$', color="tab:blue", alpha=0.5) # structure function from enstrophy dissipation
        ax[0].plot(l[:-1], S3_kolm_curved[:-1], label=r'$-\frac{6\varepsilon}{\sin^3(l)}\int_0^l \sin^3(\lambda)\,\mathrm{d}\lambda$', color="tab:blue", alpha=0.5)
        # forcing scale
        ax[0].axvline(x=np.pi/ellf, color='grey', alpha=0.2, ls='dotted')
        ax[0].text(np.pi/ellf*1.05, pp.S3_min, r'$l_f$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # axis
        ax[0].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],[r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        ax[0].set_xlim(0, np.pi)
        ax[0].set_ylim(pp.S3_min, pp.S3_max)
        # labels
        ax[0].set_xlabel(r'$l$')
        ax[0].legend(loc='lower left')
        #plt.xscale('log')
        #plt.yscale('log')
        # title
        ax[0].set_title('3rd order longitudinal structure function')
        # grid
        ax[0].grid(alpha=0.5)

        # compensated representation
        # compensate with E_diss*l
        ax[1].plot(l[1:-1], S3[1:-1] / (-3/2 * E_diss * l[1:-1]), label=r'$\frac{S^3_\parallel}{-3/2 \varepsilon l}$', color="tab:blue")
        # compensate with F_diss*l^3
        ax[1].plot(l[1:-1], S3[1:-1] / (-1/8 * F_diss * (l[1:-1])**3), label=r'$\frac{S^3_\parallel}{-1/8 \eta l^3}$', color="tab:blue", ls='dotted')
        # reference line
        ax[1].axhline(y=1, color='tab:blue', alpha=0.5)
        # forcing scale
        ax[1].axvline(x=np.pi/ellf, color='grey', alpha=0.5, ls='dotted')
        ax[1].text(np.pi/ellf*1.05, 1e-2, r'$l_f$', color='grey', fontsize=10, ha='left', va='bottom', alpha=0.4)
        # axis
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_ylim(1e-2, 3)
        # labels
        ax[1].set_xlabel(r'$l$')
        ax[1].legend(loc='upper left')
        # title
        ax[1].set_title("Compensated")
        # grid
        ax[1].grid(True, which='both', alpha=0.5)

        #plt.show()

        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/structure_function_3/S3_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)



# plot a single S2 frame
def plot_S2(run_id, frame_num):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

        # read S2
        try: # try to read data
            S2 = file['structure_function']['S2'][f"{frame_num:05d}"][()] # read S2
            S2_kolm_flat = pp.beta * file['structure_function']['S2_kolm_flat'][f"{frame_num:05d}"][()] # read Kolmogorovs predicted S2 and multiply by beta (constant prefactor)
            #S2_from_spec = file['structure_function']['S2_from_spec'][f"{frame_num:05d}"][()] # read S2 calculated via hankel transform from spectrum
        except KeyError: # if data does not exist (not all frames are processed) -> skip this function
            return

        # read length
        l = file['structure_function']['length'][()]

        # other information
        time = file['time'][f"{frame_num:05d}"][()] # simulation time of this frame


        # -- plotting --

        # size
        fig = plt.figure(figsize=(pp.Sp_width, pp.Sp_height))
        # data
        plt.plot(l[:-1], S2[:-1], label=r'$S^{2}_{\parallel} = \langle {\delta u_\parallel}^2 \rangle$', color="tab:blue") # structure function from velocity field
        plt.plot(l[:-1], S2_kolm_flat[:-1], label=r'$S^{2}_{\parallel} = \beta (\varepsilon l)^{2/3}$', color='tab:blue', alpha=0.5) # just for comparison Kolmogorovs 2/3 law, factor is arbitrarily chosen (S2 = \beta*(\epsilon*l)^{2/3} im inertial subrange, see p.97)
        #plt.plot(l[:-1], S2_from_spec[:-1], label=r'$S^{2}_{\parallel} = 4\int_0^\infty E(q) \left[\frac{1}{2}-\frac{J_1(ql)}{ql}\right]\,\mathrm{d}q$', color='tab:blue', alpha=0.5, ls='dotted')
        plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],[r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        plt.xlim(0, np.pi)
        plt.ylim(pp.S2_min, pp.S2_max)
        # labels
        plt.xlabel(r'$l$')
        plt.legend(loc='lower left')
        #plt.xscale('log')
        #plt.yscale('log')
        # title
        plt.title('2nd order longitudinal structure function')
        # grid
        plt.grid(alpha=0.5)

        plt.show()

        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/structure_function_2/S2_{frame_num:05d}.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)






# placeholder









# __________ EVOLUTIONS __________

# plot time evolution of energy, enstrophy and palinstrophy
def plot_time_evolution(run_id):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

        # check if data exists
        if "evolutions" not in file:
            raise KeyError("Group 'evolutions' not found in HDF5 file, needed for time evolution plot")
        if "E_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/E_evol' not found, needed for time evolution plot")
        if "F_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/F_evol' not found, needed for time evolution plot")
        if "P_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/P_evol' not found, needed for time evolution plot")

        # evolutions
        E_evol = file['evolutions']['E_evol'][()]
        F_evol = file['evolutions']['F_evol'][()]
        P_evol = file['evolutions']['P_evol'][()]
        # time
        time = file['evolutions']['time'][()]


        # -- plotting --

        # plot with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(pp.evolutions_width, pp.evolutions_height))

        # energy evolution
        ax1.grid(alpha=0.5)
        ax1.plot(time, E_evol, label=r'$E(t)$', color="tab:blue") # energy
        ax1.set_ylim(0, 1.3*np.max(E_evol))
        ax1.set_xlabel(r'time $t$', fontsize=12)
        ax1.set_ylabel(r'energy $\tilde{E}$', fontsize=12)
        ax1.set_title('total kinetic energy', fontsize=14)

        # enstrophy evolution
        ax2.grid(alpha=0.5)
        ax2.plot(time, F_evol, label=r'$F(t)$', color="tab:red") # enstrophy  
        ax2.set_ylim(0, 1.3*np.max(F_evol)) 
        ax2.set_xlabel(r'time $t$', fontsize=12)
        ax2.set_ylabel(r'enstrophy $\tilde{\Omega}$', fontsize=12)
        ax2.set_title('total enstrophy', fontsize=14)

        # palinstrophy evolution
        ax3.grid(alpha=0.5)
        ax3.plot(time, P_evol, label=r'$P(t)$', color="tab:green") # palinstrophy
        ax3.set_ylim(0, 1.3*np.max(P_evol))
        ax3.set_xlabel(r'time $t$', fontsize=12)
        ax3.set_ylabel(r'palinstrophy $\tilde{P}$', fontsize=12)
        ax3.set_title('total palinstrophy', fontsize=14)

        # improve layout
        fig.tight_layout()

        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/evolution/evol.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)




def plot_balance_evolution(run_id):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, mode='r') as file: # open hdf5 file

        # -- data --

        # check if data exists
        if "evolutions" not in file:
            raise KeyError("Group 'evolutions' not found in HDF5 file, needed for balance evolution plot")
        if "Ediss_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/Ediss_evol' not found, needed for balance evolution plot")
        if "Fdiss_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/Fdiss_evol' not found, needed for balance evolution plot")
        if "Ef_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/Ef_evol' not found, needed for balance evolution plot")
        if "Ff_evol" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/Ff_evol' not found, needed for balance evolution plot")
        if "Ef_evol_theo" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/Ef_evol_theo' not found, needed for balance evolution plot")
        if "Ff_evol_theo" not in file["evolutions"]:
            raise KeyError("Dataset 'evolutions/Ff_evol_theo' not found, needed for balance evolution plot")

        # evolution of dissipation
        Ediss_evol = file['evolutions']['Ediss_evol'][()]
        Fdiss_evol = file['evolutions']['Fdiss_evol'][()]

        # evolution of forcing
        Ef_evol = file['evolutions']['Ef_evol'][()]
        Ff_evol = file['evolutions']['Ff_evol'][()]

        # theoretical forcing
        Ef_evol_theo = file['evolutions']['Ef_evol_theo'][()]
        Ff_evol_theo = file['evolutions']['Ff_evol_theo'][()]

        # time
        time = file['evolutions']['time'][()]


        # -- plotting --

        # plot with three subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(pp.bal_evolutions_width, pp.bal_evolutions_height))

        # plot energy balance evolution
        ax1.grid(alpha=0.5)
        ax1.plot(time, Ediss_evol, color='tab:red', label = r'$2\kappa\tilde{\Omega}$')
        ax1.plot(time, Ef_evol, color='tab:red', label = r'$\tilde{E}^f$', alpha = 0.5)
        ax1.plot(time, Ef_evol_theo, color='tab:red', label = r'$E^f$', alpha = 0.5, ls='dotted')
        ax1.set_ylim(0, 1.3*np.max([np.max(Ediss_evol), np.max(Ef_evol)]))
        ax1.set_xlabel(r'time $t$', fontsize=12)
        ax1.legend(loc='lower left')
        ax1.set_title('total kinetic energy balance', fontsize=14)

        # plot enstrophy balance evolution
        ax2.grid(alpha=0.5)
        ax2.plot(time, Fdiss_evol, color='tab:green', label = r'$2\kappa\tilde{P}$')
        ax2.plot(time, Ff_evol, color='tab:green', label = r'$\tilde{F}^f$', alpha = 0.5)
        ax2.plot(time, Ff_evol_theo, color='tab:green', label = r'$F^f$', alpha = 0.5, ls='dotted')
        ax2.set_ylim(0, 1.3*np.max([np.max(Fdiss_evol), np.max(Ff_evol)]))
        ax2.set_xlabel(r'time $t$', fontsize=12)
        ax2.legend(loc='lower left')
        ax2.set_title('total enstrophy balance', fontsize=14)

        # improve layout
        fig.tight_layout()

        # save
        fig.savefig(pp.run_dir + f"/run_{run_id:05d}/results/evolution/evolb.png", dpi=300)

        # close figure to save RAM
        plt.close(fig)





# __________ OTHER HELPFUL FUNCTIONS __________


# find a list of all frame IDs from a hdf5 post-processed file
def find_frame_IDs(run_id):

    path = pp.run_dir + f"/run_{run_id:05d}/processed_data/run_{run_id:05d}.h5"

    with h5py.File(path, "r") as file:
        
        time_group = file["time"]

        # read all dataset names
        frame_0ids = list(time_group.keys())

        frame_ids = [int(frame_0id) for frame_0id in frame_0ids]

        return frame_ids  