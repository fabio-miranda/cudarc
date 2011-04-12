#include "defines.h"

void run(float* p_kernelTime, float* p_overheadTime, int depthPeelPass, float* p_eyePos, float* probeboxmin, float* probeboxmax, int handleTexIntersect);
void deleteGPUTextures();
void init(GLuint p_handleTexIntersect, GLuint p_handlePboOutput);
void update(int p_blocksizex, int p_blocksizey, int p_winsizex, int p_winsizey, bool p_debug0, bool p_debug1, bool p_debug2, float p_maxedge, int p_interpoltype, int p_numquadpoints, int p_numsteps, int p_nummaxcp, int p_numtraverses, int p_numelem, bool p_isosurface, bool p_volumetric, bool p_probebox);
void createGPUAdjTex(int index, int size, float* data);
void createGPUCollisionTex(int fi, int size, float* data);
void createGPUInterpolFuncTex(int index, int size, float* data);
void createGPUColorScaleTex(int numValues, int size, float* volcolorscaledata, float* isocolorscale);
void createGPUIsoControlPointsTex(int numValues, int size, float* data);
void createGPUVolControlPointsTex(int numValues, int size, float* data);
void createGPUZetaPsiGammaTex(int numValues, int size, float* data);
bool isSupported();
#ifdef CUDARC_GRADIENT_PERVERTEX
void createGPUGradientVertexTex(int fi, int size, float* data);
#endif 
void printInfoGPUMemory();
void printDebugTexture(int x, int y);