# masters_thesis_turbulence

The code presented here is used to simulate ideal Navier-Stokes turbulence in a two-dimensional flat square domain with periodic boundaries and on the 2-sphere. It is build on the open-source framework Dedalus.
For each geometry, the code is divided into simulation, post-processing and plotting. For the simulation, parameters must be specified in 'parameters.py' and 'simulation.py' must be executed. This generates an hdf5 file that stores raw data. The data can be evaluated by executing 'processing.py', where processing parameters can be specified in 'pro_parameters.py'. A standard routine for visualising the results is implemented in 'plotting.py'.
