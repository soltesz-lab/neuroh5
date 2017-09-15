// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file serialize_edge.hh
///
///  Functions for serializing edge data.
///
///  Copyright (C) 2017 Project Neuroh5.
//==============================================================================

#include <mpi.h>
#include <cstring>
#include <vector>
#include <map>
#include <string>

#include "neuroh5_types.hh"

namespace neuroh5
{
  namespace data
  {
    void serialize_edge_map (const edge_map_t& edge_map, 
                             size_t &num_packed_edges,
                             vector<char> &sendbuf);
    
    
    void serialize_rank_edge_map (const size_t num_ranks,
                                  const size_t start_rank,
                                  const rank_edge_map_t& prj_rank_edge_map, 
                                  size_t &num_packed_edges,
                                  vector<int>& sendcounts,
                                  vector<char> &sendbuf,
                                  vector<int> &sdispls);

    void deserialize_rank_edge_map (const size_t num_ranks,
                                    const vector<char> &recvbuf,
                                    const vector<int>& recvcounts,
                                    const vector<int>& rdispls,
                                    edge_map_t& prj_edge_map,
                                    size_t& num_unpacked_nodes,
                                    size_t& num_unpacked_edges
                                    );
    
    void deserialize_edge_map (const vector<char> &recvbuf,
                               edge_map_t& prj_edge_map,
                               size_t& num_unpacked_nodes,
                               size_t& num_unpacked_edges
                               );

  }
}
