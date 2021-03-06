// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file compute_vertex_metrics.hh
///
///  Function definitions for vertex metrics.
///
///  Copyright (C) 2016 Project NeuroH5.
//==============================================================================


#ifndef COMPUTE_VERTEX_METRICS_HH
#define COMPUTE_VERTEX_METRICS_HH

#include "neuroh5_types.hh"


#include <mpi.h>
#include <hdf5.h>

#include <map>
#include <vector>

namespace neuroh5
{
  namespace graph
  {
  /// @brief Computes in-degree of the vertices in the specified projections
  ///
  /// @param comm          MPI communicator
  ///
  /// @param file_name     Input file name
  ///
  /// @param prj_names     Vector of projection names to be read
  ///
  /// @param io_size       Number of I/O ranks (those ranks that conduct I/O operations)
  ///
  /// @return              HDF5 error code
    
    int compute_vertex_indegree
    (
     MPI_Comm comm,
     const std::string& input_file_name,
     const std::vector< std::pair<std::string, std::string> > prj_names,
     const size_t io_size
     );
    
    int compute_vertex_outdegree
    (
     MPI_Comm comm,
     const std::string& input_file_name,
     const std::vector< std::pair<std::string, std::string> > prj_names,
     const size_t io_size
     );

  }
  
}

#endif
