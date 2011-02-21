/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* zetapsigamma.h: load zeta psi gamma files
*/


#ifndef CUDARC_ZETAPSIGAMMA
#define CUDARC_ZETAPSIGAMMA

#define NUM_HEADER_LINES 0

/**
* 2D Psi gamma tables (for tetrahedral meshes)
*/
#define PSIGAMMA_4_FILENAME "psigamma_4.dat"
#define PSIGAMMA_16_FILENAME "psigamma_16.dat"
#define PSIGAMMA_32_FILENAME "psigamma_32.dat"
#define PSIGAMMA_64_FILENAME "psigamma_64.dat"
#define PSIGAMMA_128_FILENAME "psigamma_128.dat"
#define PSIGAMMA_256_FILENAME "psigamma_256.dat"
#define PSIGAMMA_512_FILENAME "PsiGammaTable512.txt"
#define PSIGAMMA_1024_FILENAME "psigamma_1024.dat"

/**
* 3D zeta tables (for hexahedral meshes)
*/
#define ZETA_64_FILENAME "zeta_64.dat"
#define ZETA_128_FILENAME "zeta_128.dat"
#define ZETA_256_FILENAME "zeta_256.dat"

/**
* 3D psigamma1 tables (for hexahedral meshes)
*/
#define PSIGAMMA1_64_FILENAME "psigamma1_64.dat"
#define PSIGAMMA1_128_FILENAME "psigamma1_128.dat"
#define PSIGAMMA1_256_FILENAME "psigamma1_256.dat"

/**
* 3D psigamma2 tables (for hexahedral meshes)
*/
#define PSIGAMMA2_64_FILENAME "psigamma2_64.dat"
#define PSIGAMMA2_128_FILENAME "psigamma2_128.dat"
#define PSIGAMMA2_256_FILENAME "psigamma2_256.dat"

/**
* 3D psigamma3 tables (for hexahedral meshes)
*/
#define PSIGAMMA3_64_FILENAME "psigamma3_64.dat"
#define PSIGAMMA3_128_FILENAME "psigamma3_128.dat"
#define PSIGAMMA3_256_FILENAME "psigamma3_256.dat"

/**
* Loads a psigamma or zetapsigamma table, given the path to the dir containing the files
*/
void PsiGamma(const char* zetapsigammapath, int size, float* data);
void ZetaPsiGamma(const char* zetapsigammapath, int size, float* data);

#endif