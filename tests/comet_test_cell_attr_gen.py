from mpi4py import MPI
from neuroh5.io import population_ranges, NeuroH5CellAttrGen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "rank = ", rank

#g = NeuroH5CellAttrGen("data/DGC_forest_attr_test_20170614.h5", "GC", io_size=comm.size)
forest_path = "/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/DGC_forest_syns_compressed_20180425.h5"
synapse_namespace = 'Synapse Attributes'
io_size=128
cache_size=1

it = NeuroH5CellAttrGen(forest_path, 'GC', namespace=synapse_namespace, \
                        comm=comm, io_size=io_size, cache_size=cache_size)

for (gid, synapse_dict) in it:
    if i is not None:
        print 'rank %i: gid = %i' % (rank, i)
#    print (i, e)

