
#include <mpi.h>

#include <cassert>
#include <vector>
#include <map>

#include "neuroh5_types.hh"

#include "attr_map.hh"
#include "pack_tree.hh"

using namespace std;

#define MAX_ATTR_NAME_LEN 128

namespace neuroh5
{

  namespace mpi
  {
    
    /***************************************************************************
     * Prepare MPI packed data structures with attributes for a given tree.
     **************************************************************************/

    void pack_size_index
    (
     MPI_Comm comm,
     int &sendsize
     )
    {
      int packsize=0;
      assert(MPI_Pack_size(1, MPI_CELL_IDX_T, comm, &packsize) == MPI_SUCCESS);

      sendsize += packsize;
    }

    void pack_size_attr_values
    (
     MPI_Comm comm,
     const MPI_Datatype mpi_type,
     const size_t num_elems,
     int &sendsize
     )
    {
      int packsize=0;
      if (num_elems > 0)
        {
          assert(MPI_Pack_size(num_elems, mpi_type, comm, &packsize) == MPI_SUCCESS);
          sendsize += packsize;
        }
    }
  
    void pack_index
    (
     MPI_Comm comm,
     const CELL_IDX_T index,
     const int &sendbuf_size,
     vector<uint8_t> &sendbuf,
     int &sendpos
     )
    {
      assert(MPI_Pack(&index, 1, MPI_CELL_IDX_T, &sendbuf[0], sendbuf_size, &sendpos, comm)
             == MPI_SUCCESS);
    }
  

    int pack_tree
    (
     MPI_Comm comm,
     const CELL_IDX_T &index,
     const neurotree_t &tree,
     int &sendpos,
     vector<uint8_t> &sendbuf
     )
    {
      int ierr = 0;
      int packsize=0, sendsize = 0;

      int rank, size;
      assert(MPI_Comm_size(comm, &size) >= 0);
      assert(MPI_Comm_rank(comm, &rank) >= 0);

      const vector<SECTION_IDX_T> &src_vector = get<1>(tree);
      const vector<SECTION_IDX_T> &dst_vector = get<2>(tree);
      const vector<SECTION_IDX_T> &sections = get<3>(tree);
      const vector<COORD_T> &xcoords = get<4>(tree);
      const vector<COORD_T> &ycoords = get<5>(tree);
      const vector<COORD_T> &zcoords = get<6>(tree);
      const vector<REALVAL_T> &radiuses = get<7>(tree);
      const vector<LAYER_IDX_T> &layers = get<8>(tree);
      const vector<PARENT_NODE_IDX_T> &parents = get<9>(tree);
      const vector<SWC_TYPE_T> &swc_types = get<10>(tree);

      // index size
      pack_size_index(comm, sendsize);
      // topology, section, attr size
      assert(MPI_Pack_size(3, MPI_UINT32_T, comm, &packsize) == MPI_SUCCESS);
      sendsize += packsize;
      // topology vector sizes
      pack_size_attr_values(comm, MPI_SECTION_IDX_T, src_vector.size(), sendsize);
      pack_size_attr_values(comm, MPI_SECTION_IDX_T, dst_vector.size(), sendsize);
      pack_size_attr_values(comm, MPI_SECTION_IDX_T, sections.size(), sendsize);
      // coordinate sizes 
      pack_size_attr_values(comm, MPI_COORD_T, xcoords.size(), sendsize);
      pack_size_attr_values(comm, MPI_COORD_T, ycoords.size(), sendsize);
      pack_size_attr_values(comm, MPI_COORD_T, zcoords.size(), sendsize);
      // radius sizes
      pack_size_attr_values(comm, MPI_REALVAL_T, radiuses.size(), sendsize);
      // layer size
      pack_size_attr_values(comm, MPI_LAYER_IDX_T, layers.size(), sendsize);
      // parent node size
      pack_size_attr_values(comm, MPI_PARENT_NODE_IDX_T, parents.size(), sendsize);
      // SWC type size
      pack_size_attr_values(comm, MPI_SWC_TYPE_T, swc_types.size(), sendsize);
      
      sendbuf.resize(sendbuf.size() + sendsize);

      int sendbuf_size = sendbuf.size();

      // Create MPI_PACKED object with all the tree data
      pack_index(comm, index, sendbuf_size, sendbuf, sendpos);
      vector<uint32_t> data_sizes;
      data_sizes.push_back(src_vector.size());
      data_sizes.push_back(sections.size());
      data_sizes.push_back(xcoords.size());
      assert(MPI_Pack(&data_sizes[0], data_sizes.size(), MPI_UINT32_T,
                      &sendbuf[0], sendbuf_size, &sendpos, comm) == MPI_SUCCESS);
      pack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, src_vector, sendbuf_size, sendbuf, sendpos);
      pack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, dst_vector, sendbuf_size, sendbuf, sendpos);
      pack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, sections, sendbuf_size, sendbuf, sendpos);
      // coordinate 
      pack_attr_values<COORD_T>(comm, MPI_COORD_T, xcoords, sendbuf_size, sendbuf, sendpos);
      pack_attr_values<COORD_T>(comm, MPI_COORD_T, ycoords, sendbuf_size, sendbuf, sendpos);
      pack_attr_values<COORD_T>(comm, MPI_COORD_T, zcoords, sendbuf_size, sendbuf, sendpos);
      // radius
      pack_attr_values<REALVAL_T>(comm, MPI_REALVAL_T, radiuses, sendbuf_size, sendbuf, sendpos);
      // layer
      pack_attr_values<LAYER_IDX_T>(comm, MPI_LAYER_IDX_T, layers, sendbuf_size, sendbuf, sendpos);
      // parent node
      pack_attr_values<PARENT_NODE_IDX_T>(comm, MPI_PARENT_NODE_IDX_T, parents, sendbuf_size, sendbuf, sendpos);
      // SWC type
      pack_attr_values<SWC_TYPE_T>(comm, MPI_SWC_TYPE_T, swc_types, sendbuf_size, sendbuf, sendpos);

      
      return ierr;
    }


    int pack_rank_tree_map
    (
     MPI_Comm comm,
     const map <rank_t, map<CELL_IDX_T, neurotree_t> >& rank_tree_map,
     vector<int>& sendcounts, 
     vector<int>& sdispls, 
     int &sendpos,
     vector<uint8_t> &sendbuf
     )
    {
      int srank, ssize; rank_t rank, size;
      assert(MPI_Comm_size(comm, &ssize) >= 0);
      assert(MPI_Comm_rank(comm, &srank) >= 0);
      assert(srank >= 0);
      assert(ssize > 0);
      rank = srank;
      size = ssize;

      sendcounts.resize(size);
      sdispls.resize(size);
      
      vector<int> rank_sequence;
      // Recommended all-to-all communication pattern: start at the current rank, then wrap around;
      // (as opposed to starting at rank 0)
      for (size_t dst_rank = rank; dst_rank < size; dst_rank++)
        {
          rank_sequence.push_back(dst_rank);
        }
      for (size_t dst_rank = 0; dst_rank < rank; dst_rank++)
        {
          rank_sequence.push_back(dst_rank);
        }
      
      for (const size_t& dst_rank : rank_sequence)
        {
          auto it1 = rank_tree_map.find(dst_rank); 
          sdispls[dst_rank] = sendpos;
          
          if (it1 != rank_tree_map.end())
            {
              for (auto it2 = it1->second.cbegin(); it2 != it1->second.cend(); ++it2)
                {
                  CELL_IDX_T  idx   = it2->first;
                  const neurotree_t &tree = it2->second;
                  
                  pack_tree(comm, idx, tree, sendpos, sendbuf);
                }
            }
          sendcounts[dst_rank] = sendpos - sdispls[dst_rank];
        }

      return 0;
    }

    
    void unpack_index
    (
     MPI_Comm comm,
     CELL_IDX_T &index,
     const size_t &recvbuf_size,
     const vector<uint8_t> &recvbuf,
     int &recvpos
     )
    {
      assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                        &index, 1, MPI_CELL_IDX_T, comm) == MPI_SUCCESS);
    }

    neurotree_t unpack_tree
    (
     MPI_Comm comm,
     const vector<uint8_t> &recvbuf,
     int &recvpos
     )
    {
      
      CELL_IDX_T index;
      vector<SECTION_IDX_T> src_vector;
      vector<SECTION_IDX_T> dst_vector;
      vector<SECTION_IDX_T> sections;
      vector<COORD_T> xcoords;
      vector<COORD_T> ycoords;
      vector<COORD_T> zcoords;
      vector<REALVAL_T> radiuses;
      vector<LAYER_IDX_T> layers;
      vector<PARENT_NODE_IDX_T> parents;
      vector<SWC_TYPE_T> swc_types;

      size_t recvbuf_size = recvbuf.size();
      
      unpack_index(comm, index, recvbuf_size, recvbuf, recvpos);
      vector<uint32_t> data_sizes(3);
      assert(MPI_Unpack(&recvbuf[0], recvbuf_size, &recvpos,
                        &data_sizes[0], 3, MPI_UINT32_T, comm) == MPI_SUCCESS);

      unpack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, data_sizes[0], src_vector, recvbuf_size, recvbuf, recvpos);
      unpack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, data_sizes[0], dst_vector, recvbuf_size, recvbuf, recvpos);
      unpack_attr_values<SECTION_IDX_T>(comm, MPI_SECTION_IDX_T, data_sizes[1], sections, recvbuf_size, recvbuf, recvpos);
      // coordinate 
      unpack_attr_values<COORD_T>(comm, MPI_COORD_T, data_sizes[2], xcoords, recvbuf_size, recvbuf, recvpos);
      unpack_attr_values<COORD_T>(comm, MPI_COORD_T, data_sizes[2], ycoords, recvbuf_size, recvbuf, recvpos);
      unpack_attr_values<COORD_T>(comm, MPI_COORD_T, data_sizes[2], zcoords, recvbuf_size, recvbuf, recvpos);
      // radius
      unpack_attr_values<REALVAL_T>(comm, MPI_REALVAL_T, data_sizes[2], radiuses, recvbuf_size, recvbuf, recvpos);
      // layer
      unpack_attr_values<LAYER_IDX_T>(comm, MPI_LAYER_IDX_T, data_sizes[2], layers, recvbuf_size, recvbuf, recvpos);
      // parent node
      unpack_attr_values<PARENT_NODE_IDX_T>(comm, MPI_PARENT_NODE_IDX_T, data_sizes[2], parents, recvbuf_size, recvbuf, recvpos);
      // SWC type
      unpack_attr_values<SWC_TYPE_T>(comm, MPI_SWC_TYPE_T, data_sizes[2], swc_types, recvbuf_size, recvbuf, recvpos);

      neurotree_t tree = make_tuple(index, src_vector, dst_vector, sections,
                                    xcoords, ycoords, zcoords,
                                    radiuses, layers, parents,
                                    swc_types);

      
      return tree;
    }
    
    int unpack_tree_map
    (
     MPI_Comm comm,
     const vector<uint8_t> &recvbuf,
     int &recvpos,
     map<CELL_IDX_T, neurotree_t> &tree_map
     )
    {
      while ((size_t)recvpos < recvbuf.size())
        {
          neurotree_t tree = unpack_tree(comm, recvbuf, recvpos);
          tree_map.insert(make_pair(get<0>(tree), tree));
          assert((size_t)recvpos <= recvbuf.size());
        }

      return 0;
    }

    int unpack_tree_vector
    (
     MPI_Comm comm,
     const vector<uint8_t> &recvbuf,
     int &recvpos,
     vector<neurotree_t> &tree_vector
     )
    {
      while ((size_t)recvpos < recvbuf.size())
        {
          neurotree_t tree = unpack_tree(comm, recvbuf, recvpos);
          tree_vector.push_back(tree);
          assert((size_t)recvpos <= recvbuf.size());
        }

      return 0;
    }

  }
}
