#ifndef VALIDATE_SELECTION_EDGE_LIST_HH
#define VALIDATE_SELECTION_EDGE_LIST_HH

#include "neuroh5_types.hh"

#include <set>
#include <vector>

namespace neuroh5
{
  namespace graph
  {
    /// @brief Validates that each edge in a projection has source and
      ///        destination indices that are within the population ranges
      ///        defined for that projection
      ///
      /// @param src_start     Updated with global starting index of source
      ///                      population
      ///
      /// @param dst_start     Updated with global starting index of destination
      ///                      population
      ///
      /// @param selection_dst_idx       Destination Index (starting destination index for
      ///                      each entry in dst_ptr)
      ///
      /// @param selection_dst_ptr       Destination Pointer (pointer to Source Index
      ///                      where the source indices for a given destination
      ///                      can be located)
      ///
      /// @param src_idx       Source Index (source indices of edges)
      ///
      /// @param pop_ranges    Map where the key is the starting index of a
      ///                      population, and the value is the number of nodes
      ///                      (vertices) and population index, filled by this
      ///                      procedure
      ///
      /// @param pop_pairs     Set of source/destination pairs, filled by this
      ///                      procedure
      ///
      /// @return              True if the edges are valid, false otherwise
      extern bool validate_selection_edge_list
      (
       const NODE_IDX_T&                          src_start,
       const NODE_IDX_T&                          dst_start,
       const std::vector<NODE_IDX_T>&             selection_dst_idx,
       const std::vector<DST_PTR_T>&              selection_dst_ptr,
       const std::vector<NODE_IDX_T>&             src_idx,
       const pop_search_range_map_t&              pop_search_ranges,
       const std::set< std::pair<pop_t, pop_t> >& pop_pairs
       );
  }
}

#endif
