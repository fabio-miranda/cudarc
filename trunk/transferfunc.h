/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* transferfunc.h: load transfer function files
* File format (transfer function):
* Line 1: type of transfer function (constant, discrete or linear)
* Line 2: number of control points
* Other lines: r g b a (0..1)
*
* File format (iso values):
* Line 1: number of iso values
* Other lines: scalar r g b a (0..1)
*/


#ifndef CUDARC_TRANSFERFUNC
#define CUDARC_TRANSFERFUNC

#include <topsview/colorscale/colorscale.h>

/**
* Loads a transfer function file, given the path to the file.
* Ignores lines beggining with #
*/
TpvColorScale* TransferFunc(const char* tffilepath, float scalarmin, float scalarmax);
//TpvColorScale* IsoValues(const char* isofilepath, int texsize, float scalarmin, float scalarmax);

#endif