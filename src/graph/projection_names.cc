// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file projection_names.cc
///
///  Functions for reading projection names from an HDF5 file.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================
#include <mpi.h>
#include <hdf5.h>

#include <cstring>
#include <vector>

#include "debug.hh"
#include "path_names.hh"
#include "group_contents.hh"
#include "serialize_data.hh"
#include "throw_assert.hh"

#define MAX_PRJ_NAME 1024

using namespace std;

namespace neuroh5
{
    namespace graph
    {
      //////////////////////////////////////////////////////////////////////////
      herr_t iterate_cb
      (
       hid_t             grp,
       const char*       name,
       const H5L_info_t* info,
       void*             op_data
       )
      {
        vector<string>* ptr = (vector<string>*)op_data;
        ptr->push_back(string(name));
        return 0;
      }

      //////////////////////////////////////////////////////////////////////////
      herr_t read_projection_names
      (
       MPI_Comm             comm,
       const std::string&   file_name,
       vector<pair<string,string>>&      prj_names
       )
      {
        herr_t ierr = 0;

        int rank, size;

        throw_assert_nomsg(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
        throw_assert_nomsg(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);
        
        vector<string> prj_src_pop_names, prj_dst_pop_names;
        
        // MPI rank 0 reads and broadcasts the number of ranges
        hid_t file = -1;

        // MPI rank 0 reads and broadcasts the projection names
        if (rank == 0)
          {
            vector <string> dst_pop_names;
            file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file >= 0)
              {

                throw_assert_nomsg(hdf5::group_contents(comm, file, hdf5::PROJECTIONS, dst_pop_names) >= 0);
                
                for (size_t i=0; i<dst_pop_names.size(); i++)
                  {
                    vector <string> src_pop_names;
                    const string& dst_pop_name = dst_pop_names[i];
                    
                    throw_assert_nomsg(hdf5::group_contents(comm, file, hdf5::PROJECTIONS+"/"+dst_pop_name, src_pop_names) >= 0);
                    
                    for (size_t j=0; j<src_pop_names.size(); j++)
                      {
                        prj_src_pop_names.push_back(src_pop_names[j]);
                        prj_dst_pop_names.push_back(dst_pop_name);
                      }
                  }
                
                throw_assert_nomsg(H5Fclose(file) >= 0);
              }
            else
              {
                ierr = file;
              }
          }
        
        // Broadcast projection names
        {
          vector<char> sendbuf; uint32_t sendbuf_size=0;
          if (rank == 0)
            {
              data::serialize_data(prj_src_pop_names, sendbuf);
              sendbuf_size = sendbuf.size();
            }

          throw_assert_nomsg(MPI_Bcast(&sendbuf_size, 1, MPI_UINT32_T, 0, comm) == MPI_SUCCESS);
          sendbuf.resize(sendbuf_size);
          throw_assert_nomsg(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, 0, comm) == MPI_SUCCESS);
          
          if (rank != 0)
            {
              data::deserialize_data(sendbuf, prj_src_pop_names);
            }
        }
        {
          vector<char> sendbuf; uint32_t sendbuf_size=0;
          if (rank == 0)
            {
              data::serialize_data(prj_dst_pop_names, sendbuf);
              sendbuf_size = sendbuf.size();
            }
          

          throw_assert_nomsg(MPI_Bcast(&sendbuf_size, 1, MPI_UINT32_T, 0, comm) == MPI_SUCCESS);
          sendbuf.resize(sendbuf_size);
          throw_assert_nomsg(MPI_Bcast(&sendbuf[0], sendbuf_size, MPI_CHAR, 0, comm) == MPI_SUCCESS);
          
          if (rank != 0)
            {
              data::deserialize_data(sendbuf, prj_dst_pop_names);
            }
        }


        for (size_t i=0; i<prj_dst_pop_names.size(); i++)
          {
            prj_names.push_back(make_pair(prj_src_pop_names[i],
                                          prj_dst_pop_names[i]));
          }

        return ierr;
      }
    
    }
}
