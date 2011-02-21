/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* interpolfunc.h: compute the interpolation function of the elements, using SVD
*/

#ifndef CUDARC_GRADIENT_H
#define CUDARC_GRADIENT_H

#include <alg/matrix.h>

/**
* Function that receives an array of position and scalars, and returns the computed interpolation function
*/
void ComputeGradientLeastSquares(int size, float* positions, float* scalars, float* gradient);

/**
* Called by ComputeGradientLeastSquares. Given A and B, returns x (Ax = b)
*/
int LeastSquares(int numlines, int numcols, long double** A, long double** b, long double* x);

/**
* Multiply two matrices
*/
long double** Multiply(int a_m, int a_n, long double** a, int b_m, int b_n, long double** b);

/**
* Calculate the transpose of a matrix
*/
long double** Transpose(int a_m, int a_n, long double** a);

/**
 * Computes the svd of matrix a[m][n].
 * A = U*W*Vt
 * U replaces A on output
 * From Numerical Recipes in C: The Art of Scientific Computing
 */
void svdcmp(long double **a, int m, int n, long double w[], long double **v);

#endif