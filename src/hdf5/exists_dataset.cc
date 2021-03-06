
#include "hdf5.h"
#include <string>
#include <vector>

#include "tokenize.hh"
#include "exists_dataset.hh"

using namespace std;

namespace neuroh5
{
  namespace hdf5
  {
      
    /*****************************************************************************
     * Check if dataset exists
     *****************************************************************************/
    int exists_dataset
    (
     hid_t  loc,
     const string& path
     )
    {
      const string delim = "/";
      herr_t status=-1;  string ppath;
      vector<string> tokens;
      data::tokenize (path, delim, tokens);
      
      for (string value : tokens)
        {
          if (ppath.empty())
            {
              ppath = value;
            }
          else
            {
              ppath = ppath + delim + value;
            }
          status = H5Lexists (loc, ppath.c_str(), H5P_DEFAULT);
          if (status <= 0)
            break;
        }

      return status;
    }
  }
}
