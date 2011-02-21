/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* cudarc.h: handles all ray casting computations, and calls to the CUDA kernels
*/

#ifndef CUDA_RC_H
#define CUDA_RC_H

#include <topsview/geometry/tetrageometry3.h>
#include <topsview/colorscale/colorscale.h>
#include <shader/glslshader.h>
#include <ugl/fbo.h>
#include <ugl/pbuffer.h>
#include <ugl/texture.h>
#include <ufont/fontmanager.h>
#include <gpos/model/modelnew.h>
#include <gpos/model/geometry.h>
#include <explodedview/explodedmodel.h>

class Time {
public:
  Time(){}
  Time(int p_maxNumPeel){
    //maxNumPeel = p_maxNumPeel;

    totalKcTime = 0;
    totalOvTime = 0;
    totalShaderTime = 0;

    //numFrames = p_numFrames;
    currentFrame = 0;
  }
  void Reset(){
    //if(currentFrame == numFrames){
    currentFrame = 0;

    lastOvTime = totalOvTime;
    totalOvTime = 0;

    lastShaderTime = totalShaderTime;
    totalShaderTime = 0;

    lastKcTime = totalKcTime;
    totalKcTime = 0;
    //}
  }
  void SetTime(int index, float p_shaderTime, float p_kernelCallTime, float p_overheadTime, int p_numFrags){

    totalKcTime += p_kernelCallTime;

    totalOvTime += p_overheadTime;

    totalShaderTime += p_shaderTime;
  }
  //int numFrames;
  int currentFrame;
  int numPeel;

  double totalShaderTime;
  double lastShaderTime;

  float totalKcTime;
  float lastKcTime;

  float totalOvTime;
  float lastOvTime;

};

struct MemoryInfo{
  int numElem;
  int numAdjTex;
  int sizeAdjTex;
  int numFaces;
  int numNodes;
  int sizeCollisionTex;
  int numInterpolFuncTex;
  int sizeInterpolFuncTex;
  int numValuesTf;
  int sizeTf;
  int numValuesZetaPsiGamma;
  int sizeZetaPsiGamma;
};


MODEL_CLASS_TEMPLATE
class CudaRC{

public:
  CudaRC();
  ~CudaRC();

  /**
  * Set paths to the .cg shader files, and the zeta psi gamma table
  */
  void SetPaths(int zetapsigammasize, const char* zetapsigammapath, const char* shaderpath);

  /**
  * Set geometry, for oil reservoir models or fem models
  */
  void SetGeometry(TpvTetraGeometry3<MODELCLASS>* tetrageometry);
  void SetResGeometry(ResGeometry* resGeometry);

  /**
  * Set tops model, for oil reservoir or fem models
  */
  void SetModel(TopModel* topmodel);

  /**
  * Set a multitops model, for oil reservoir or fem models.
  * If a TopMultiModel is used, we will only use the first level
  */
  void SetModel(TopMultiModel* topmultimodel);

  /**
  * Set volumetric color scale, for oil reservoir models or fem models
  */
  void SetVolumetricColorScale(TpvColorScale* colorScale);

  /**
  * Set iso color scale, for oil reservoir models or fem models
  */
  void SetIsoColorScale(TpvColorScale* colorScale);

  /**
  * Set property, for oil reservoir models or fem models
  */
  void SetProperty(TpvProperty* prop);

  /**
  * Set if integration is normalized by the max edge length
  */
  void SetInterpolationType(int interpoltype);
  void SetNumSteps(int numsteps);
  void SetNumTraverses(int numtraverses);
  void SetNumPeeling(int numpeel);

  /**
  * Set constant integration
  */
  void SetMaxEdgeLengthEnabled(bool enabled); 

  void SetDebug(bool flag);
  void SetIsoSurface(bool flag);
  void SetVolumetric(bool flag);
  void SetExplodedView(bool flag);
  void SetBlockSize(int sizex, int sizey);
  void SetWindowSize(int sizex, int sizey);
  void SetMaxNumPeel(int maxpeel);
  void SetTessellation(int tessellation);



  /**
  * Create textures and upload them to the GPU
  */
  void BuildTextures();

  /**
  * Compute the boundary elements, for the rays initial positions
  */
  void ComputeBdryFaces();


  /**
  * Render callback
  */
  void Render(bool bdryonly, float* eyePos, float* eyeDir, float* eyeUp, float eyeZNear, float eyeFov, bool debug);

  /**
  * Returns the output of the CUDA computation
  */
  GLuint GetPboOutputId();

  /**
  * Display info regarding the ray casting
  */
  void DisplayInfo(double totaltime);
  void PrintInfo();
  void PrintTime();

  /**
  * Set explosion factor of the model, for exploded view
  */
  void SetExplosionFactor(float explosionfactor);

  Time m_time;

private:

  /**
  * Update methods
  */
  void Update();
  void UpdateWindow();
  void UpdateGeometry();
  void UpdateColorScale();
  void UpdateProperty();
  void UpdateZetaPsiGamma();


  /**
  * Render only one pass of the model, for depth peeling purposes
  */
  void RenderSinglePass(float* eyePos);

  /**
  * Compute the boundary elements, for the rays initial positions
  */
  void ComputeTetraBdryFaces();
  void ComputeHexaBdryFaces();

  /**
  * Build textures
  */
  void BuildHexaMeshTextures(float** nodesData, float** adjacenciesData);
  void BuildTetraMeshTextures(float** nodesData, float** adjacenciesData);
  void BuildHexaInterpolFuncTexture(float** interpolfuncdata);
  void BuildHexaScalarTexture(float** scalarData);
  void BuildTetraInterpolFuncTexture(float** interpolfuncdata);
  void BuildTetraScalarTexture(float** scalarData);
  void BuildZetaPsiGammaTexture(float* psiGammaData);
  void BuildColorScaleTexture(float* volcolorscalesata, float* isocolorscaledata);
  void BuildControlPointsTexture(float* cpdata, TpvColorScale* colorscale);
  void BuildAmbOcclusionTexture(float** ambocclusdata);

  //ExplodedModel* m_explodedmodel;

  TpvTetraGeometry3<MODELCLASS>* m_tetraGeometry;
  ResGeometry* m_resGeometry;

  //Fem and Res models
  TopModel* m_topmodel;
  TpvProperty* m_property;
  TpvColorScale* m_volcolorscale;
  TpvColorScale* m_isocolorscale;

  //Array with the display lists (one for each exploded part)
  GLuint* m_bdryDispList;
  GLuint m_handlePboOutput;
  GLuint m_handleOccQuery;

  //Depth peeling variables
  UGLTexture* m_uglTexIntersect;
  UGLTexture* m_uglDepthIntersect;
  UGLFrameBuffer* m_uglFbIntersect;
  GLSLShader* m_shaderIntersect;


  AlgVector m_winsize;
  AlgVector m_blocksize;
  int m_maxNumPeel;
  int* m_elemId;
  int* m_diffnodesperelem;
  float m_explosionfactor;
  bool m_maxedgeenabled;
  bool m_debug;
  bool m_isosurface;
  bool m_volumetric;
  bool m_explodedview;
  int m_interpoltype;
  int m_numsteps;
  int m_numtraverses;
  int m_numpeeling;
  int m_zetapsigammasize;
  int m_tfsize;
  int m_tessellation;
  const char* m_zetapsigammapath;
  const char* m_shaderpath;

  bool m_cuda_initialized;
  bool m_update_cuda;
  bool m_initialized;
  bool m_update_geometry;
  bool m_update_colorscale;
  bool m_update_property;
  bool m_update_window;
  bool m_update_zetapsigamma;
  bool m_update_memoryinfo;
  bool m_update_maxnumpeel;

  MemoryInfo m_memoryInfo;
};

#include "cudarc.inl"

#endif