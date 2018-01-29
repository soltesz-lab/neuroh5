from mpi4py import MPI
from neuroh5.io import read_graph, scatter_read_graph

comm = MPI.COMM_WORLD
print "rank = ", comm.Get_rank()
print "size = ", comm.Get_size()

g = scatter_read_graph("data/dentate_test.h5")
print (g)

