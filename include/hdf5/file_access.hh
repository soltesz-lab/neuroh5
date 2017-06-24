#ifndef FILE_ACCESS_HH
#define FILE_ACCESS_HH

#include "hdf5.h"

#include <vector>

namespace neuroh5
{
  namespace hdf5
  {
    /*****************************************************************************
     * Routines for opening and closing NeuroH5 file for reading and writing.
     *****************************************************************************/
    
    hid_t open_file
    (
     MPI_Comm comm,
     const std::string& file_name,
     const bool collective = false,
     const bool rdwr = false
     );

    herr_t close_file
    (
     hid_t &file
     );

    
  }
}

#endif
