// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file read_graph.cc
///
///  Top-level functions for reading graphs in DBS (Destination Block Sparse)
///  format.
///
///  Copyright (C) 2016-2021 Project NeuroH5.
//==============================================================================

#include "debug.hh"

#include "edge_attributes.hh"

#include "read_projection.hh"
#include "cell_populations.hh"
#include "read_graph.hh"
#include "mpi_debug.hh"
#include "throw_assert.hh"

using namespace neuroh5::data;
using namespace std;

namespace neuroh5
{
  namespace graph
  {
    int read_graph
    (
     MPI_Comm              comm,
     const std::string&    file_name,
     const vector<string>& edge_attr_name_spaces,
     const vector< pair<string, string> >& prj_names,
     std::vector<edge_map_t>& prj_vector,
     vector < map <string, vector < vector<string> > > > & edge_attr_names_vector,
     size_t&              total_num_nodes,
     size_t&              local_num_edges,
     size_t&              total_num_edges
     )
    {
      int status = 0;
      int rank, size;
      throw_assert_nomsg(MPI_Comm_size(comm, &size) == MPI_SUCCESS);
      throw_assert_nomsg(MPI_Comm_rank(comm, &rank) == MPI_SUCCESS);
      
      // read the population info
      pop_label_map_t pop_labels;
      pop_range_map_t pop_ranges;
      set< pair<pop_t, pop_t> > pop_pairs;
      throw_assert_nomsg(cell::read_population_combos(comm, file_name, pop_pairs) >= 0);
      throw_assert_nomsg(cell::read_population_ranges
             (comm, file_name, pop_ranges, total_num_nodes) >= 0);
      throw_assert_nomsg(cell::read_population_labels(comm, file_name, pop_labels) >= 0);

      pop_search_range_map_t pop_search_ranges;
      for (auto &x : pop_ranges)
        {
          pop_search_ranges.insert(make_pair(x.second.start, make_pair(x.second.count, x.first)));
        }

      // read the edges
      for (size_t i = 0; i < prj_names.size(); i++)
        {
          size_t local_prj_num_nodes;
          size_t local_prj_num_edges;
          size_t total_prj_num_edges;
          hsize_t local_read_blocks;
          hsize_t total_read_blocks;

          //printf("Task %d reading projection %lu (%s)\n", rank, i, prj_names[i].c_str());

          string src_pop_name = prj_names[i].first, dst_pop_name = prj_names[i].second;
          pop_t dst_pop_idx = 0, src_pop_idx = 0;
          bool src_pop_set = false, dst_pop_set = false;
      
          for (auto &x : pop_labels)
            {
              if (src_pop_name == get<1>(x))
                {
                  src_pop_idx = get<0>(x);
                  src_pop_set = true;
                }
              if (dst_pop_name == get<1>(x))
                {
                  dst_pop_idx = get<0>(x);
                  dst_pop_set = true;
                }
            }
          throw_assert_nomsg(dst_pop_set && src_pop_set);

          NODE_IDX_T dst_start = pop_ranges[dst_pop_idx].start;
          NODE_IDX_T src_start = pop_ranges[src_pop_idx].start;

          mpi::MPI_DEBUG(comm, "read_graph: src_pop_name = ", src_pop_name,
                         " dst_pop_name = ", dst_pop_name,
                         " dst_start = ", dst_start,
                         " src_start = ", src_start);

          throw_assert_nomsg(graph::read_projection
                 (comm, file_name, pop_search_ranges, pop_pairs,
                  src_pop_name, dst_pop_name, 
                  src_start, dst_start, edge_attr_name_spaces, 
                  prj_vector, edge_attr_names_vector,
                  local_prj_num_nodes,
                  local_prj_num_edges, total_prj_num_edges,
                  local_read_blocks, total_read_blocks) >= 0);

          mpi::MPI_DEBUG(comm, "read_graph: projection ", i, " has a total of ", total_prj_num_edges, " edges");
          
          total_num_edges = total_num_edges + total_prj_num_edges;
          local_num_edges = local_num_edges + local_prj_num_edges;
        }

      size_t sum_local_num_edges = 0;
      status = MPI_Reduce(&local_num_edges, &sum_local_num_edges, 1,
                          MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
      throw_assert_nomsg(status == MPI_SUCCESS);
      
      if (rank == 0)
        {
          if (sum_local_num_edges != total_num_edges)
            {
              printf("sum_local_num_edges = %lu total_num_edges = %lu\n",
                     sum_local_num_edges, total_num_edges);
            }
          throw_assert_nomsg(sum_local_num_edges == total_num_edges);
        }

      return 0;
    }




  }
}
