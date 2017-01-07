// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file population_reader.cc
///
///  Functions for reading population information and validating the
///  source and destination indices of edges.
///
///  Copyright (C) 2016 Project Neurograph.
//==============================================================================

#include "debug.hh"
#include "population_reader.hh"

#include "hdf5_path_names.hh"

#include <cstdio>
#include <iostream>
#include <vector>

#undef NDEBUG
#include <cassert>

using namespace std;
using namespace ngh5::model;
using namespace ngh5::io::hdf5;

namespace ngh5
{
  namespace io
  {
    namespace hdf5
    {
      /*************************************************************************
       * Read the valid population combinations
       *************************************************************************/

      herr_t read_population_combos
      (
       MPI_Comm                   comm,
       const std::string&         file_name,
       set< pair<pop_t,pop_t> >&  pop_pairs
       )
      {
        herr_t ierr = 0;

        int rank, size;

        assert(MPI_Comm_size(comm, &size) >= 0);
        assert(MPI_Comm_rank(comm, &rank) >= 0);

        // MPI rank 0 reads and broadcasts the number of pairs

        uint64_t num_pairs;

        hid_t file = -1, dset = -1;

        // process 0 reads the number of pairs and broadcasts
        if (rank == 0)
          {
            file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file >= 0);

            dset = H5Dopen2(file, h5types_path_join(POP_COMB).c_str(),
                            H5P_DEFAULT);
            assert(dset >= 0);
            hid_t fspace = H5Dget_space(dset);
            assert(fspace >= 0);
            num_pairs = (uint64_t) H5Sget_simple_extent_npoints(fspace);
            assert(num_pairs > 0);
            assert(H5Sclose(fspace) >= 0);
          }

        assert(MPI_Bcast(&num_pairs, 1, MPI_UINT64_T, 0, comm) >= 0);

        // allocate buffers
        vector<pop_t> v(2*num_pairs);

        // MPI rank 0 reads and broadcasts the population pairs

        if (rank == 0)
          {
            vector<pop_comb_t> vpp(num_pairs);
            hid_t ftype = H5Dget_type(dset);
            assert(ftype >= 0);
            hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);

            assert(H5Dread(dset, mtype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                           &vpp[0]) >= 0);
            for (size_t i = 0; i < vpp.size(); ++i)
              {
                v[2*i]   = vpp[i].src;
                v[2*i+1] = vpp[i].dst;
              }

            assert(H5Tclose(mtype) >= 0);
            assert(H5Tclose(ftype) >= 0);

            assert(H5Dclose(dset) >= 0);
            assert(H5Fclose(file) >= 0);
          }

        assert(MPI_Bcast(&v[0], (int)2*num_pairs, MPI_UINT16_T, 0, comm) >= 0);

        // populate the set
        pop_pairs.clear();
        for (size_t i = 0; i < v.size(); i += 2)
          {
            pop_pairs.insert(make_pair(v[i], v[i+1]));
          }

        return ierr;
      }

      /*************************************************************************
       * Read the population ranges
       *************************************************************************/

      herr_t read_population_ranges
      (
       MPI_Comm                                comm,
       const std::string&                      file_name,
       map<NODE_IDX_T, pair<uint32_t,pop_t> >& pop_ranges,
       vector<pop_range_t> &pop_vector,
       size_t &n_nodes
       )
      {
        herr_t ierr = 0;

        int rank, size;
        assert(MPI_Comm_size(comm, &size) >= 0);
        assert(MPI_Comm_rank(comm, &rank) >= 0);

        // MPI rank 0 reads and broadcasts the number of ranges

        uint64_t num_ranges;

        hid_t file = -1, dset = -1;

        // process 0 reads the number of ranges and broadcasts
        if (rank == 0)
          {
            file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file >= 0);
            dset = H5Dopen2(file, h5types_path_join(POP).c_str(), H5P_DEFAULT);
            assert(dset >= 0);
            hid_t fspace = H5Dget_space(dset);
            assert(fspace >= 0);
            num_ranges = (uint64_t) H5Sget_simple_extent_npoints(fspace);
            assert(num_ranges > 0);
            assert(H5Sclose(fspace) >= 0);
          }

        assert(MPI_Bcast(&num_ranges, 1, MPI_UINT64_T, 0, comm) >= 0);

        // allocate buffers
        pop_vector.resize(num_ranges);

        // MPI rank 0 reads and broadcasts the population ranges

        if (rank == 0)
          {
            hid_t ftype = H5Dget_type(dset);
            assert(ftype >= 0);
            hid_t mtype = H5Tget_native_type(ftype, H5T_DIR_ASCEND);

            assert(H5Dread(dset, mtype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                           &pop_vector[0]) >= 0);

            assert(H5Tclose(mtype) >= 0);
            assert(H5Tclose(ftype) >= 0);

            assert(H5Dclose(dset) >= 0);
            assert(H5Fclose(file) >= 0);
          }

        assert(MPI_Bcast(&pop_vector[0], (int)num_ranges*sizeof(pop_range_t),
                         MPI_BYTE, 0, comm) >= 0);

        n_nodes = 0;
        for(size_t i = 0; i < pop_vector.size(); ++i)
          {
            pop_ranges.insert(make_pair(pop_vector[i].start,
                                        make_pair(pop_vector[i].count,
                                                  pop_vector[i].pop)));
            n_nodes = n_nodes + pop_vector[i].count;
          }

        return ierr;
      }
    }
  }

}