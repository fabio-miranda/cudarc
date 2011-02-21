/**
 * Fabio Markus Miranda
 * fmiranda@tecgraf.puc-rio.br
 * fabiom@gmail.com
 * Dec 2010
 * 
 * cudamemory.h: CUDA memory variables
 */

#ifndef CUDARC_MEMORY_H
#define CUDARC_MEMORY_H

#include "defines.h"

#include <cuda_runtime_api.h>
#include <cutil_inline.h>

//Host memory
cudaGraphicsResource* cudaPboHandleOutput;
cudaGraphicsResource* cudaTexHandleIntersect;


//Device memory
//CUDA does not support an array of textures, so we have to declare a large number of textures.
//If we chose to use 2d textures, the maximum size of the texture would decrease.

/**
 * Hex. and tet.
 */
texture<float4, 1, cudaReadModeElementType> texAdj0;
texture<float4, 1, cudaReadModeElementType> texInterpolFunc0;
texture<float4, 1, cudaReadModeElementType> texAmbOcclusionFunc;
texture<float4, 2, cudaReadModeElementType> texIntersect;
texture<float4, 1, cudaReadModeElementType> texVolumetricColorScale;
texture<float4, 1, cudaReadModeElementType> texIsoColorScale;
texture<float4, 1, cudaReadModeElementType> texVolumetricControlPoints;
texture<float4, 1, cudaReadModeElementType> texIsoControlPoints;


#ifdef CUDARC_HEX
/**
* Hex. only
*/
#ifdef CUDARC_BILINEAR
texture<float4, 1, cudaReadModeElementType> texNode0;
texture<float4, 1, cudaReadModeElementType> texNode1;
texture<float4, 1, cudaReadModeElementType> texNode2;
texture<float4, 1, cudaReadModeElementType> texNode3;
texture<float4, 1, cudaReadModeElementType> texNode4;
texture<float4, 1, cudaReadModeElementType> texNode5;
texture<float4, 1, cudaReadModeElementType> texNode6;
texture<float4, 1, cudaReadModeElementType> texNode7;
#else
texture<float4, 1, cudaReadModeElementType> texFace0Eq;
texture<float4, 1, cudaReadModeElementType> texFace1Eq;
texture<float4, 1, cudaReadModeElementType> texFace2Eq;
texture<float4, 1, cudaReadModeElementType> texFace3Eq;
texture<float4, 1, cudaReadModeElementType> texFace4Eq;
texture<float4, 1, cudaReadModeElementType> texFace5Eq;
#endif
texture<float4, 1, cudaReadModeElementType> texAdj1;
texture<float4, 1, cudaReadModeElementType> texInterpolFunc1;
texture<float, 2, cudaReadModeElementType> texZetaPsiGamma;
static float* dev_collision[8];
static float* dev_adj[2];
static float* dev_interpolfunc[2];
#else

/**
 * Tet. only
 */
texture<float, 2, cudaReadModeElementType> texZetaPsiGamma;
static float* dev_collision[4];
static float* dev_adj[1];
static float* dev_interpolfunc[1];
static float* dev_gradientVertex[4];
#ifdef CUDARC_PLUCKER
texture<float4, 1, cudaReadModeElementType> texNode0;
texture<float4, 1, cudaReadModeElementType> texNode1;
texture<float4, 1, cudaReadModeElementType> texNode2;
texture<float4, 1, cudaReadModeElementType> texNode3;
#else
texture<float4, 1, cudaReadModeElementType> texFace0Eq;
texture<float4, 1, cudaReadModeElementType> texFace1Eq;
texture<float4, 1, cudaReadModeElementType> texFace2Eq;
texture<float4, 1, cudaReadModeElementType> texFace3Eq;
#endif
#endif
#endif


texture<float4, 1, cudaReadModeElementType> texGrad0;
texture<float4, 1, cudaReadModeElementType> texGrad1;
texture<float4, 1, cudaReadModeElementType> texGrad2;
texture<float4, 1, cudaReadModeElementType> texGrad3;


static float4* dev_outputData;
cudaArray* dev_intersectData;
//(tex1d can only be used on cudaArray memory, not linear memory: http://forums.nvidia.com/index.php?showtopic=164023)
//linear memory can only be used using tex1dfetch. Or use a 2dtexture (see link for more info.)
cudaArray* dev_volcolorscale; 
cudaArray* dev_isocolorscale;
cudaArray* dev_volcontrolpoints;
cudaArray* dev_isocontrolpoints;
cudaArray* dev_zetaPsiGamma;

struct ConstMemory{
  float numTets;
  float2 screenSize;
  float interpoltype;
  float maxedge;
  float debug;
  float numsteps;
  float numtraverses;
  float isosurface;
  float volumetric;
};


__constant__ ConstMemory constMemory;

//Host memory
static int sizex, sizey;
dim3 grids;
dim3 threads;

//Debug
static float4 *dev_debug;
static float4 debug [512*512];