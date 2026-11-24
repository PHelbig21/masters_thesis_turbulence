# calculation
import numpy as np
import dedalus.public as d3  # to create dedalus fields

# parallelisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
import hashlib # for _stable_seed



Nphi = 32
Ntheta = 16
R = 1
dealias = 3/2
dtype = np.float64



coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype) 


# ---- spectral grids ---- TODO

# find global shape
shape = basis.global_shape(grid_space=(False, False), scales=(1, 1))
# find global dedalus spectral grid indices
i_s = np.arange(shape[0])
j_s = np.arange(shape[1])
# create global spectral ij-grid
i_grid, j_grid = np.meshgrid(i_s, j_s, indexing='ij')
# find slicing to extract local spectral grid
field = dist.Field(name='field', bases=basis)
local_slices = field.domain.dist.coeff_layout.slices(field.domain, scales=1)
local_slice = local_slices[0]
i_grid = i_grid[local_slice]
j_grid = j_grid[local_slice]

# spectral grid (grid indexing internally used by Dedalus)
l_grid, m_grid = basis.elements_to_groups((False, False), (i_grid, j_grid))

l_grid_thinned = l_grid[::2]
m_grid_thinned = m_grid[::2]


#print(l_grid)
#print(m_grid)

print(l_grid_thinned)
print(m_grid_thinned)

