from mpi4py import MPI

import h5py


rank = MPI.COMM_WORLD.rank
f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
dset = f.create_dataset('test', (MPI.COMM_WORLD.size,), dtype='i')
dset[rank] = rank
f.close()
