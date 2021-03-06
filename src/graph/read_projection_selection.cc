// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_projection_selection.cc
///
///  Functions for reading edge information in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2020 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "neuroh5_types.hh"
#include "read_projection_selection.hh"
#include "edge_attributes.hh"
#include "read_projection_dataset_selection.hh"
#include "validate_selection_edge_list.hh"
#include "append_edge_map_selection.hh"
#include "throw_assert.hh"
#include "mpi_debug.hh"

#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace neuroh5
{
  namespace graph
  {
    
    /**************************************************************************
     * Reads a subset of the basic DBS graph structure
     *************************************************************************/

    herr_t read_projection_selection
    (
     MPI_Comm                   comm,
     const std::string&         file_name,
     const pop_search_range_map_t& pop_search_ranges,
     const set < pair<pop_t, pop_t> >& pop_pairs,
     const std::string&         src_pop_name,
     const std::string&         dst_pop_name,
     const NODE_IDX_T&          src_start,
     const NODE_IDX_T&          dst_start,
     const vector<string>&      attr_namespaces,
     const std::vector<NODE_IDX_T>&  selection,
     vector<edge_map_t>&       prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&                    local_num_nodes,
     size_t&                    local_num_edges,
     size_t&                    total_num_edges,
     bool collective
     )
    {
      herr_t ierr = 0;
      int rank, size;
      throw_assert(MPI_Comm_size(comm, &size) == MPI_SUCCESS, "unable to obtain MPI communicator size");
      throw_assert(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS, "unable to obtain MPI communicator rank");

      
      DST_PTR_T edge_base, edge_count;
      vector<NODE_IDX_T> selection_dst_idx;
      vector<DST_PTR_T> selection_dst_ptr;
      vector<NODE_IDX_T> src_idx;
      map<string, data::NamedAttrVal> edge_attr_map;
      vector< pair<hsize_t,hsize_t> > src_idx_ranges;
        
      mpi::MPI_DEBUG(comm, "read_projection_selection: ", src_pop_name, " -> ", dst_pop_name, " : "
                     "selection of size ", selection.size());
      throw_assert(hdf5::read_projection_dataset_selection(comm, file_name, src_pop_name, dst_pop_name,
                                                           src_start, dst_start, selection, edge_base,
                                                           selection_dst_idx, selection_dst_ptr, src_idx_ranges,
                                                           src_idx, total_num_edges) >= 0,
                   "error in read_projection_dataset_selection");
      
      mpi::MPI_DEBUG(comm, "read_projection_selection: ", src_pop_name, " -> ", dst_pop_name, " :", 
                     " size of destination index is ", selection_dst_idx.size(),
                     " size of destination pointer is ", selection_dst_ptr.size(),
                     "; total_num_edges is ", total_num_edges);
      mpi::MPI_DEBUG(comm, "read_projection_selection: validating projection ", src_pop_name, " -> ", dst_pop_name);
      
      // validate the edges
      throw_assert(validate_selection_edge_list(src_start, dst_start, selection_dst_idx,
                                                selection_dst_ptr, src_idx, pop_search_ranges, pop_pairs) ==
                   true,
                   "error in validating edge list"); 
      
      edge_count = src_idx.size();
      local_num_edges = edge_count;

      map <string, vector < vector<string> > > edge_attr_names;
      for (string attr_namespace : attr_namespaces) 
        {
          vector< pair<string,AttrKind> > edge_attr_info;
          
          throw_assert(graph::get_edge_attributes(comm, file_name, src_pop_name, dst_pop_name,
                                                  attr_namespace, edge_attr_info) >= 0,
                       "unable to obtain edge attributes");
          
          throw_assert(graph::read_all_edge_attribute_selection
                       (comm, file_name, src_pop_name, dst_pop_name, attr_namespace,
                        edge_base, edge_count, selection_dst_idx, selection_dst_ptr, src_idx_ranges,
                        edge_attr_info, edge_attr_map[attr_namespace]) >= 0,
                       "error in read_all_edge_attribute_selection");

          edge_attr_map[attr_namespace].attr_names(edge_attr_names[attr_namespace]);
        }
      
      size_t local_prj_num_edges=0;

      edge_map_t prj_edge_map;
      // append to the vectors representing a projection (sources,
      // destinations, edge attributes)
      throw_assert(data::append_edge_map_selection(dst_start, src_start, selection_dst_idx, selection_dst_ptr,
                                                   src_idx, attr_namespaces, edge_attr_map,
                                                   local_prj_num_edges, prj_edge_map,
                                                   EdgeMapDst) >= 0,
                   "error in append_edge_map_selection");
      local_num_nodes = prj_edge_map.size();

      mpi::MPI_DEBUG(comm, "read_projection_selection: ", src_pop_name, " -> ", dst_pop_name, " :", 
                     " local_num_nodes =  ", local_num_nodes);
      
      // ensure that all edges in the projection have been read and
      // appended to edge_list
      throw_assert(local_prj_num_edges == edge_count,
                   "mismatch in projection edge count");

      prj_vector.push_back(prj_edge_map);
      edge_attr_names_vector.push_back (edge_attr_names);
      
      
      return ierr;
    }

    
  }
}

