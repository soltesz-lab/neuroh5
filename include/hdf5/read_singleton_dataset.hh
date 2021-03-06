#ifndef READ_SINGLETON_DATASET_HH
#define READ_SINGLETON_DATASET_HH

#include <hdf5.h>
#include <mpi.h>

#include <string>

#include "throw_assert.hh"
namespace neuroh5
{
  namespace hdf5
  {
    template <class T>
    void read_singleton_dataset
    (
     MPI_Comm comm,
     const std::string& file_name,
     const std::string& path,
     hid_t              mem_type,
     MPI_Datatype       mpi_type,
     T&                 result
     )
    {
      int rank=0, size;

      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS, "error in MPI_Comm_size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS, "error in MPI_Comm_rank");

      if (rank == 0)
        {
          hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          throw_assert(file >= 0, "error in H5Fopen");

          hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
          throw_assert(dset >= 0, "error in H5Dopen");
          throw_assert(H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &result) >= 0,
                 "error in H5Dread");
          throw_assert(H5Dclose(dset) >= 0, "error in H5Dclose");

          throw_assert(H5Fclose(file) >= 0, "error in H5Fclose");

        }

      throw_assert(MPI_Bcast(&result, 1, mpi_type, 0, comm) == MPI_SUCCESS,
                   "error in MPI_Bcast");
    }
  }
}

#endif
