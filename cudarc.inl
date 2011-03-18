/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* cudarc.inl: handles all ray casting computations, and calls to the CUDA kernels
*/

#include "defines.h"
#include "cudarc.h"
#include "interpolfunc.h"
#include "zetapsigamma.h"

//TODO: rerun vissetup and export defines.h interface.h
//#include "../../../src/explodedview/defines.h"
//#include "../../../src/explodedview/interface.h"

#include <GL/glew.h>
#include <GL/glext.h>
#include <gpos/model/model.h>
#include <gpos/model/geometry.h>
#include <gpos/model/modelnew.h>
#include <topsview/geometry/geometry3.h>
#include <alg/matrix.h>
#include <tops/model.h>
#include <tops/iterators.h>
#include <topsview/geometry/triangleset.h>
#include <topsview/geometry/geometryutil.h>
#include <topsview/geometry/array.h>
#include <topsview/geometry/tetrageometry3.h>
#include <topsview/geometry/tetrahedronset.h>
#include <topsview/geometry/bdryfaces.h>
#include <topsview/colorscale/colorscale.h>
#include <topsview/defines.h>
#include <alg/vector.h>
#include <ufont/glutmessage.h>
#include <ugl/ugl.h>

#include <stdlib.h>


extern "C" void run(float* p_kernelTime, float* p_overheadTime, int depthPeelPass, float* p_eyePos, float* probeboxmin, float* probeboxmax, int handleTexIntersect, float delta, float deltaW, float zero);
extern "C" void deleteGPUTextures();
extern "C" void init(GLuint p_handleTexIntersect, GLuint p_handlePboOutput);
extern "C" void update(int p_blocksizex, int p_blocksizey, int p_winsizex, int p_winsizey, bool p_debug, float p_maxedge, int p_interpoltype, int p_numsteps, int p_numtraverses, int p_numelem, bool p_isosurface, bool p_volumetric, bool p_probebox);
extern "C" void createGPUAdjTex(int index, int size, float* data);
extern "C" void createGPUCollisionTex(int fi, int size, float* data);
extern "C" void createGPUInterpolFuncTex(int index, int size, float* data);
extern "C" void createGPUColorScaleTex(int numValues, int size, float* volcolorscaledata, float* isocolorscale);
extern "C" void createGPUIsoControlPointsTex(int numValues, int size, float* data);
extern "C" void createGPUVolControlPointsTex(int numValues, int size, float* data);
extern "C" void createGPUZetaPsiGammaTex(int numValues, int size, float* data);
#ifdef CUDARC_GRADIENT_PERVERTEX
extern "C" void createGPUGradientVertexTex(int fi, int size, float* data);
#endif 
extern "C" void printInfoGPUMemory();



MODEL_CLASS_TEMPLATE
CudaRC<MODELCLASS>::CudaRC(){

  m_resGeometry = NULL;
  m_volcolorscale = NULL;
  m_isocolorscale = NULL;
  m_isovalues = NULL;
  m_explosionfactor = 1.0;
  m_tfsize = 1024;
  m_tessellation = 1;
  m_maxedgeenabled = true;
  m_debug = false;
  m_isosurface = false;
  m_volumetric = true;
  m_probeboxenabled = false;
  m_normalizedfield = true;
  m_uglFbIntersect = new UGLFrameBuffer[2];
  m_uglTexIntersect = new UGLTexture[2];
  m_uglDepthIntersect = new UGLTexture[2];
  for(int i=0; i<2; i++){
    m_uglFbIntersect[i] = UGLFrameBuffer();
    m_uglTexIntersect[i] = UGLTexture();
    m_uglDepthIntersect[i] = UGLTexture();
  }
  m_shaderIntersect = new GLSLShader(); 

  m_cuda_initialized = false;
  m_update_cuda = true;
  m_initialized = true;
  m_update_geometry = true;
  m_update_colorscale = true;
  m_update_property = true;
  m_update_window = true;
  m_update_zetapsigamma = true;
  m_update_memoryinfo = true;
  m_update_maxnumpeel = true;
}

MODEL_CLASS_TEMPLATE
CudaRC<MODELCLASS>::~CudaRC(){

  deleteGPUTextures();
  delete m_shaderIntersect;

}



MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::Update(){

  if(m_update_memoryinfo){
    m_tetraGeometry->SetField(m_property);
    m_tetraGeometry->Update();

    //Textures size
    m_memoryInfo = MemoryInfo();

#ifdef CUDARC_HEX
    //m_memoryInfo.numElem = m_resGeometry->Nact();
    m_memoryInfo.numElem = m_topmodel->GetNElem();
    m_memoryInfo.numAdjTex = 2;
    m_memoryInfo.numFaces = 6;
    m_memoryInfo.numNodes = 8;    
    m_memoryInfo.numInterpolFuncTex = 2;
    m_memoryInfo.numValuesZetaPsiGamma = m_zetapsigammasize;
    m_memoryInfo.sizeZetaPsiGamma = m_zetapsigammasize * m_zetapsigammasize * sizeof(float);
#else
    m_memoryInfo.numElem = m_tetraGeometry->GetTetrahedra()->GetSize();
    m_memoryInfo.numAdjTex = 1;
    m_memoryInfo.numFaces = 4;
    m_memoryInfo.numNodes = 4;
    m_memoryInfo.numInterpolFuncTex = 1;
    m_memoryInfo.numValuesZetaPsiGamma = m_zetapsigammasize;
    m_memoryInfo.sizeZetaPsiGamma = m_zetapsigammasize * m_zetapsigammasize * sizeof(float);
#endif

    m_memoryInfo.numValuesTf = m_tfsize;
    m_memoryInfo.sizeTf = 4 * m_memoryInfo.numValuesTf * sizeof(float);
    m_memoryInfo.sizeCollisionTex = (m_memoryInfo.numElem + 1) * 4 * sizeof(float);
    m_memoryInfo.sizeInterpolFuncTex = (m_memoryInfo.numElem + 1) * 4 * sizeof(float);
    m_memoryInfo.sizeAdjTex = (m_memoryInfo.numElem + 1) * 4 * sizeof(float);
	m_memoryInfo.sizeGradientVertexTex = (m_memoryInfo.numElem + 1) * 4 * sizeof(float);

  }

  if(m_update_window)
    UpdateWindow();
  

  if(!m_cuda_initialized){
    init(m_uglTexIntersect[0].GetTextureId(), m_handlePboOutput);
    m_update_cuda = true;
  }

  if(m_update_cuda){
    float maxedge = m_maxedgeenabled ? m_tetraGeometry->GetMaxEdgeLength() : 1.0f;
    int numelem = m_memoryInfo.numElem;
    update(m_blocksize.x, m_blocksize.y, m_winsize.x, m_winsize.y, m_debug, maxedge, m_interpoltype, m_numsteps, m_numtraverses, numelem, m_isosurface, m_volumetric, m_probeboxenabled);
  }

  if(m_update_maxnumpeel){
#ifdef CUDARC_TIME
    //Time
    m_time = Time(m_maxNumPeel);
#endif
  }

  if(m_update_geometry){
    m_tetraGeometry->SetField(m_property);
    m_tetraGeometry->SetNormalizeField(m_normalizedfield);
    m_tetraGeometry->Update();

    float** adjacenciesData = new float*[m_memoryInfo.numAdjTex];
    for(int i=0; i<m_memoryInfo.numAdjTex; i++)
      adjacenciesData[i] = new float[4 * (m_memoryInfo.numElem + 1)];

    int size = m_memoryInfo.numFaces;
#ifdef CUDARC_BILINEAR
    size = m_memoryInfo.numNodes;
#endif

    
    float** collisionData = new float*[size];
    for(int i=0; i<size; i++){
      collisionData[i] = new float[4 * (m_memoryInfo.numElem + 1)];
    }
#ifdef CUDARC_HEX
    ComputeHexaBdryFaces();
    BuildHexaMeshTextures(collisionData, adjacenciesData);
#else
    ComputeTetraBdryFaces();
    BuildTetraMeshTextures(collisionData, adjacenciesData);
#endif
    
    for(int i=0; i<m_memoryInfo.numAdjTex; i++)
      createGPUAdjTex(i, m_memoryInfo.sizeAdjTex, adjacenciesData[i]);
    
    for(int ni=0; ni<size; ni++)
      createGPUCollisionTex(ni, m_memoryInfo.sizeCollisionTex, collisionData[ni]);
    
    
    for(int i=0; i<size; i++)
      delete collisionData[i];
    delete [] collisionData;

    for(int i=0; i<m_memoryInfo.numAdjTex; i++)
      delete [] adjacenciesData[i];
    delete [] adjacenciesData;

    //Test float precision to see if we can recover both adjid and adjfaceid using / and % operations
    float maxti = m_memoryInfo.numElem;
    float maxadjfaceid = m_memoryInfo.numFaces - 1;
    float adjacenciesdata = m_memoryInfo.numFaces * maxti + maxadjfaceid;
    int recoveredti = adjacenciesdata / m_memoryInfo.numFaces;
    int recoveredfaceid = (int)adjacenciesdata % m_memoryInfo.numFaces;
    printf("\nMax element id: %f\n", maxti);
    printf("Max face id: %f\n", maxadjfaceid);
    printf("Stored adjacency info: %f\n", adjacenciesdata);
    printf("Recovered element id: %d\n", recoveredti);
    printf("Recovered face id: %d\n", recoveredfaceid);

    assert(recoveredti == maxti);
    assert(recoveredfaceid == maxadjfaceid);
  }

  if(m_update_property){
    float** interpolFuncData = new float*[m_memoryInfo.numInterpolFuncTex];
    for(int i=0; i<m_memoryInfo.numInterpolFuncTex; i++)
      interpolFuncData[i] = new float[4 * (m_memoryInfo.numElem + 1)];

	  float** gradientVertexData = new float*[m_memoryInfo.numNodes];
	  for(int i=0; i<m_memoryInfo.numNodes; i++)
        gradientVertexData[i] = new float[4 * (m_memoryInfo.numElem + 1)];

#ifdef CUDARC_HEX
    BuildHexaInterpolFuncTexture(interpolFuncData);
#else
    BuildTetraInterpolFuncTexture(interpolFuncData);
#ifdef CUDARC_GRADIENT_PERVERTEX
	  BuildTetraVertexGradientTextures( gradientVertexData, interpolFuncData );
#endif
#endif

    for(int i=0; i<m_memoryInfo.numInterpolFuncTex; i++)
      createGPUInterpolFuncTex(i, m_memoryInfo.sizeInterpolFuncTex, interpolFuncData[i]);

    for(int i=0; i<m_memoryInfo.numInterpolFuncTex; i++)
      delete [] interpolFuncData[i];
    delete [] interpolFuncData;

#ifdef CUDARC_GRADIENT_PERVERTEX
    for(int i=0; i<m_memoryInfo.numFaces; i++)
	    createGPUGradientVertexTex(i, m_memoryInfo.sizeGradientVertexTex, gradientVertexData[i]);

    for(int i=0; i<m_memoryInfo.numFaces; i++)
      delete gradientVertexData[i];
    delete [] gradientVertexData;
#endif
  }

  if(m_update_colorscale){
    float* volcolorscalesata = new float[4 * m_memoryInfo.numValuesTf];
    float* isocolorscaledata = new float[4 * m_memoryInfo.numValuesTf];
    float* volcontronpointsdata = new float[4 * m_memoryInfo.numValuesTf];
    float* isocontronpointsdata = new float[4 * m_memoryInfo.numValuesTf];

    int volnumcp = m_volcolorscale->GetNumberOfValues();
    int isonumcp = m_isocolorscale->GetNumberOfValues();
    float* volcp = new float[volnumcp];
    float* isocp = new float[isonumcp];

    for(int i=0; i<volnumcp; i++)
      volcp[i] = m_volcolorscale->GetValue(i);

    for(int i=0; i<isonumcp; i++)
      isocp[i] = m_isocolorscale->GetValue(i);

    BuildColorScaleTexture(volcolorscalesata, isocolorscaledata);
    BuildControlPointsTexture(volcontronpointsdata, volnumcp, volcp, m_volcolorscale->GetValue(0), m_volcolorscale->GetValue(volnumcp-1));
    BuildControlPointsTexture(isocontronpointsdata, isonumcp-1, m_isovalues, m_isocolorscale->GetValue(0), m_isocolorscale->GetValue(isonumcp-1));
    createGPUColorScaleTex(m_memoryInfo.numValuesTf, m_memoryInfo.sizeTf, volcolorscalesata, isocolorscaledata);
    createGPUVolControlPointsTex(m_memoryInfo.numValuesTf, m_memoryInfo.sizeTf, volcontronpointsdata);
    createGPUIsoControlPointsTex(m_memoryInfo.numValuesTf, m_memoryInfo.sizeTf, isocontronpointsdata);
    delete [] volcolorscalesata;
    delete [] isocolorscaledata;
    delete [] volcontronpointsdata;
    delete [] isocontronpointsdata;
    delete [] volcp;
    delete [] isocp;
  }

  if(m_update_zetapsigamma){
    char vertPath[256];
    char fragPath[256];
    sprintf(vertPath, "%s", m_shaderpath);
    sprintf(vertPath+strlen(m_shaderpath), "intersect.vert");

    sprintf(fragPath, "%s", m_shaderpath);
    sprintf(fragPath+strlen(m_shaderpath), "intersect.frag");

    m_shaderIntersect->CompileFile(Shader::VP, vertPath);
    m_shaderIntersect->CompileFile(Shader::FP, fragPath);
    m_shaderIntersect->Link();

    float* zetaPsiGammaData;
#ifdef CUDARC_HEX
    //zetaPsiGammaData = new float[4 * m_memoryInfo.numValuesZetaPsiGamma * m_memoryInfo.numValuesZetaPsiGamma * m_memoryInfo.numValuesZetaPsiGamma];
    //ZetaPsiGamma(m_zetapsigammapath, m_memoryInfo.numValuesZetaPsiGamma, zetaPsiGammaData);
    zetaPsiGammaData = new float[m_memoryInfo.numValuesZetaPsiGamma * m_memoryInfo.numValuesZetaPsiGamma];
    PsiGamma(m_zetapsigammapath, m_memoryInfo.numValuesZetaPsiGamma, zetaPsiGammaData);
    createGPUZetaPsiGammaTex(m_memoryInfo.numValuesZetaPsiGamma, m_memoryInfo.sizeZetaPsiGamma, zetaPsiGammaData);
    delete [] zetaPsiGammaData;  
#else
    zetaPsiGammaData = new float[m_memoryInfo.numValuesZetaPsiGamma * m_memoryInfo.numValuesZetaPsiGamma];
    PsiGamma(m_zetapsigammapath, m_memoryInfo.numValuesZetaPsiGamma, zetaPsiGammaData);
    createGPUZetaPsiGammaTex(m_memoryInfo.numValuesZetaPsiGamma, m_memoryInfo.sizeZetaPsiGamma, zetaPsiGammaData);
    delete [] zetaPsiGammaData;  
#endif
    
  }

  m_cuda_initialized = true;
  m_update_cuda = false;
  m_initialized = true;
  m_update_geometry = false;
  m_update_colorscale = false;
  m_update_property = false;
  m_update_window = false;
  m_update_zetapsigamma = false;
  m_update_memoryinfo = false;
  m_update_maxnumpeel = false;

}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::UpdateWindow(){
  
  for(int i=0; i<2; i++){
    //Texture (for FBO)
    m_uglTexIntersect[i].SetData(m_winsize.x, m_winsize.y, GL_RGBA, GL_FLOAT, NULL, true);
    m_uglTexIntersect[i].SetInternalFormat(GL_RGBA32F_ARB);
    m_uglTexIntersect[i].Update();

    //Depth (for FBO);
    m_uglDepthIntersect[i].SetData(m_winsize.x, m_winsize.y, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL, true);
    m_uglDepthIntersect[i].Update();

    //FBO
    m_uglFbIntersect[i].SetSize(m_winsize.x, m_winsize.y);
    m_uglFbIntersect[i].SetupRenderToTexture(&m_uglTexIntersect[i], &m_uglDepthIntersect[i]);

  }

  //PBO
  glGenBuffers(1, &m_handlePboOutput);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_handlePboOutput);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, m_winsize.x * m_winsize.y * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW_ARB);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  //Occlusion query
  glGenQueries(1, &m_handleOccQuery);
  

}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::ComputeHexaBdryFaces(){

  TpvBdryFacesGeometry bdryFaces = TpvBdryFacesGeometry();
  bdryFaces.SetModel(m_topmodel);
  if(m_resGeometry != NULL) bdryFaces.SetActiveElements(m_resGeometry->GetVisibleCells());
  TpvTriangleSet* triangleSet = bdryFaces.GetTriangleSet();
  const TopFaceUse* faces = triangleSet->GetFacesTops();
  int numFaces = triangleSet->GetNumFacesTops();

  //Elements ids, excluding inactive cells
  m_elemId = new int[m_topmodel->GetNElem()+1];
  m_diffnodesperelem = new int[m_topmodel->GetNElem()+1];
  int id = 1;
  int numhexpernumnodes[9];
  for(int i=0; i<=8; i++)
    numhexpernumnodes[i] = 0;
  for(int i=1; i<=m_topmodel->GetNElem(); i++){
    TopElement el = m_topmodel->GetElemAtId(i);
    if(el.IsValid() && ((m_resGeometry == NULL) || (m_resGeometry != NULL && m_resGeometry->IsActiveCell(i)))){
      m_elemId[i] = id;

      //Find if it is degenerated
      float pos[8][3];
      for(int j=0; j<m_topmodel->GetNNodes(el); j++){
        TopNode node = m_topmodel->GetNode(el, j);
        m_topmodel->GetPosition(node, &pos[j][0], &pos[j][1], &pos[j][2]);
      }
      int numnodes = 8;
      for(int aux1=0; aux1<8; aux1++){
        for(int aux2=aux1; aux2<8; aux2++){
          if(aux1 != aux2 && pos[aux1][0] == pos[aux2][0] && pos[aux1][1] == pos[aux2][1] && pos[aux1][2] == pos[aux2][2]){
            numnodes--;
          }
        }
      }
      numhexpernumnodes[numnodes]++;
      m_diffnodesperelem[i] = numnodes;

      id++;
    }
    else{
      m_elemId[i] = 0;
    }
  }

#ifdef CUDARC_VERBOSE
  printf("Total elements: %d\n", id-1);
  for(int i=0; i<9; i++)
    printf("%d nodes hex.: %d\n", i, numhexpernumnodes[i]);
#endif

  //Render calls
  m_bdryDispList = new GLuint[1];
  m_bdryDispList[0] = glGenLists(1);
  glNewList(m_bdryDispList[0], GL_COMPILE);
  glBegin(GL_TRIANGLES);
  int count = 0;
  for(int i=0; i<numFaces; i++){
    TopElement el = m_topmodel->GetElem((faces[i]));
    int id = m_topmodel->GetId(el);

    if(el.IsValid()){
      
      //Save both the element id and the face id of the --current-- cell (not the adj)
      static int aux[6] = {1,0,3,4,2,5}; //fem
      //static int aux[6] = {0,1,2,3,4,5}; //res
      int faceid = faces[i].GetLocalId();
      int id_x = (6 * m_elemId[id] + aux[faceid]) / CUDARC_MAX_MESH_TEXTURE_WIDTH;
      int id_y = (6 * m_elemId[id] + aux[faceid]) % CUDARC_MAX_MESH_TEXTURE_WIDTH;

      //Save both the element id and the front face id
      glMultiTexCoord2f(GL_TEXTURE0, id_x, id_y);
      
      AlgVector pos[4];
      for(int j=0; j<m_topmodel->GetNNodes(faces[i]); j++){
        TopNode node = m_topmodel->GetNode(faces[i], j);
        m_topmodel->GetPosition(node, &pos[j].x, &pos[j].y, &pos[j].z);
      }

      /**
       * 0 -- 3
       * |    |
       * 1 -- 2
       */
      
      for(int ui=0; ui<m_tessellation; ui++){
        for(int vi=0; vi<m_tessellation; vi++){
          float u = ui / (float)m_tessellation;
          float v = vi / (float)m_tessellation;
          float unext = (ui+1.0f) / (float)m_tessellation;
          float vnext = (vi+1.0f) / (float)m_tessellation;
          AlgVector p0 = (1.0f - unext)*(1.0f - v) * pos[0] + (1.0f - unext) * v * pos[1] + unext * (1.0f - v) * pos[3] + unext * v * pos[2];
          AlgVector p1 = (1.0f - u)*(1.0f - v) * pos[0] + (1.0f - u) * v * pos[1] + u * (1.0f - v) * pos[3] + u * v * pos[2];
          AlgVector p2 = (1.0f - u)*(1.0f - vnext) * pos[0] + (1.0f - u) * vnext * pos[1] + u * (1.0f - vnext) * pos[3] + u * vnext * pos[2];
          AlgVector p3 = (1.0f - unext)*(1.0f - vnext) * pos[0] + (1.0f - unext) * vnext * pos[1] + unext * (1.0f - vnext) * pos[3] + unext * vnext * pos[2];

          glVertex3fv((float *)p0);
          glVertex3fv((float *)p1);
          glVertex3fv((float *)p2);
          glVertex3fv((float *)p0);
          glVertex3fv((float *)p2);
          glVertex3fv((float *)p3);

        }
      }
      
      //012, 023
      /*
      glVertex3fv((float *)pos[0]);
      glVertex3fv((float *)pos[1]);
      glVertex3fv((float *)pos[2]);
      glVertex3fv((float *)pos[0]);
      glVertex3fv((float *)pos[2]);
      glVertex3fv((float *)pos[3]);
      */
      /*
      //013, 123
      glVertex3fv(pos[0]);
      glVertex3fv(pos[1]);
      glVertex3fv(pos[3]);
      glVertex3fv(pos[1]);
      glVertex3fv(pos[2]);
      glVertex3fv(pos[3]);
      */
      /*
      AlgVector p[4];

      const static int femfaces[6][4] = {
        {4,5,6,7}, //{4,7,6,5}  {0,1,5,4} {4,5,6,7}
        {0,3,2,1}, //{0,1,2,3}  {2,3,7,6} {0,3,2,1}

        {1,2,6,5}, //{1,2,6,5}
        {0,4,7,3}, //{0,4,7,3} <-

        {0,1,5,4}, //{0,1,5,4} {4,5,6,7} {0,1,5,4} 2,6,7,3       4,5,6,7     0,1,5,4
        {2,3,7,6}  //{2,3,7,6} {0,3,2,1} {2,3,7,6} {2,6,7,3}                 2,3,7,6
      };

      m_topmodel->GetPosition(m_topmodel->GetNode(el, femfaces[faceid][0]), &(p[0].x), &(p[0].y), &(p[0].z));
      m_topmodel->GetPosition(m_topmodel->GetNode(el, femfaces[faceid][1]), &(p[1].x), &(p[1].y), &(p[1].z));
      m_topmodel->GetPosition(m_topmodel->GetNode(el, femfaces[faceid][2]), &(p[2].x), &(p[2].y), &(p[2].z));
      m_topmodel->GetPosition(m_topmodel->GetNode(el, femfaces[faceid][3]), &(p[3].x), &(p[3].y), &(p[3].z));

      glVertex3fv((float *)p[0]);
      glVertex3fv((float *)p[1]);
      glVertex3fv((float *)p[2]);
      glVertex3fv((float *)p[0]);
      glVertex3fv((float *)p[2]);
      glVertex3fv((float *)p[3]);
      */
    }
    else{
      count++;
    }
  }
  glEnd();
  glEndList();

}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::ComputeTetraBdryFaces(){

  TpvTetrahedronSet* set = m_tetraGeometry->GetTetrahedra();
  int numFaces = 4;
  static const int F_N[4][3] = {{1,2,3},{2,0,3},{3,0,1},{1,0,2}};
  TpvIntArray bdryIndices;
  bdryIndices.Resize(0);
  for(int i=0; i<m_memoryInfo.numElem; i++){
    for(int j=0; j<numFaces; j++){
      
      //if(set->GetAdj(i, j) == 0 && m_explodedmodel->GetElemPartId(i) != 1){
      if(set->GetAdj(i, j) == 0){
        bdryIndices.Append(i+1); 
        break;
      }
    }
  }

  //Create display list
  m_bdryDispList = new GLuint[1];
  m_bdryDispList[0] = glGenLists(1);
  glNewList(m_bdryDispList[0], GL_COMPILE);
  glBegin(GL_TRIANGLES);
  int aux = 0;
  for(int i=0; i<bdryIndices.Size(); i++){
    int id = bdryIndices[i];
    //unsigned short tId_x = tId%10000;
    //unsigned short tId_y = tId/10000;

    for(int j=0; j<numFaces; j++){
      //Only add the face who has no adj. tetra.
      //if(set->GetAdj(id-1, j) == 0 && m_explodedmodel->GetElemPartId(id-1) != 1)  {
      if(set->GetAdj(id-1, j) == 0)  {

        //Save both the element id and the face id of the --current-- cell (not the adj)
        float id_x = (4 * id + j) / CUDARC_MAX_MESH_TEXTURE_WIDTH;
        float id_y = (4 * id + j) % CUDARC_MAX_MESH_TEXTURE_WIDTH;
        //int id_x = (id) / CUDARC_MAX_MESH_TEXTURE_WIDTH;
        //int id_y = (id) % CUDARC_MAX_MESH_TEXTURE_HEIGHT;

        
        glMultiTexCoord2f(GL_TEXTURE0, id_x, id_y);
        
        for(int k=0; k<3; k++){
          glVertex3fv(set->GetPosition(id-1, F_N[j][k])); 
          //std::cout << tetSet->GetPosition(id-1, F_N[j][k])[0] << "\n";
        }
      }
    }
  }
  
  glEnd();
  //glutSolidSphere(10000, 100, 100);
  glEndList();
}

float ComputeHexNormal(AlgVector p0, AlgVector p1, AlgVector p2, AlgVector p3, AlgVector* normal1, AlgVector* normal2, AlgVector* coplanarvertex1, AlgVector* coplanarvertex2){

/*
 *	0 -- 2
 *  |    |
 *  1 -- 3
 */

  AlgVector v10 = (p1 - p0).Normalized();
  AlgVector v20 = (p2 - p0).Normalized();
  AlgVector v30 = (p3 - p0).Normalized();
  AlgVector v21 = (p2 - p1).Normalized();
  AlgVector v31 = (p3 - p1).Normalized();
  //AlgVector v23 = (p2 - p3).Normalized();
  float distances[4];
  AlgVector normals[4];
  AlgVector coplanarvertices[4];
  
  //
  normals[0] = (v10 ^ v30).Normalized();
  if(normals[0].Length() > 0){
    coplanarvertices[0] = p0;
    distances[0] = (p2 - coplanarvertices[0]).Dot(normals[0]);
  }

  normals[1] = (v30 ^ v20).Normalized();
  if(normals[1].Length() > 0){
    coplanarvertices[1] = p0;
    distances[1] = (p1 - coplanarvertices[1]).Dot(normals[1]);
  }

  //
  normals[2] = (v21 ^ (-v10)).Normalized();
  if(normals[2].Length() > 0){
    coplanarvertices[2] = p1;
    distances[2] = (p3 - coplanarvertices[2]).Dot(normals[2]);
  }  

  normals[3] = (v31 ^ v21).Normalized();
  if(normals[3].Length() > 0){
    coplanarvertices[3] = p1;
    distances[3] = (p0 - coplanarvertices[3]).Dot(normals[3]);
  }

  
  float nearestpoint = FLT_MAX;
  int normalid = -1;
  for(int i=0; i<4; i++){
    if(normals[i].Length() > 0 && distances[i] < nearestpoint){
      nearestpoint = distances[i];
      normalid = i;
    }
  }
  

  //Check for degenerate face
  if(normalid != -1){
    //*normal1 = normals[normalid];
    //*coplanarvertex1 = coplanarvertices[normalid];
    
    if(normalid == 0 || normalid == 1){
      *normal1 = normals[0];
      *coplanarvertex1 = coplanarvertices[0];

      *normal2 = -normals[1];
      *coplanarvertex2 = coplanarvertices[1];
    }
    else{
      *normal1 = normals[2];
      *coplanarvertex1 = coplanarvertices[2];

      *normal2 = -normals[3];
      *coplanarvertex2 = coplanarvertices[3];
    }

    return 0;
    //return abs(distances[normalid]);
  }
  else{
    *normal1 = AlgVector(0, 0, 0);
    *normal2 = AlgVector(0, 0, 0);
    return 0;
  }
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::BuildHexaMeshTextures(float** collisionData, float** adjacenciesData){

#ifdef CUDARC_VERBOSE
  printf("\nBuilding Mesh Textures... ");
#endif

  int numFaces = 6;
  int** faces = new int*[6];
  int* faceindex;

  //Fem
  const static int femfaces[6][4] = {
    {4,5,6,7}, //{4,7,6,5}  {0,1,5,4} {4,5,6,7}
    {0,3,2,1}, //{0,1,2,3}  {2,3,7,6} {0,3,2,1}

    {1,2,6,5}, //{1,2,6,5} 
    {0,4,7,3}, //{0,4,7,3} <-

    {0,1,5,4}, //{0,1,5,4} {4,5,6,7} {0,1,5,4} 2,6,7,3       4,5,6,7     0,1,5,4
    {2,3,7,6}  //{2,3,7,6} {0,3,2,1} {2,3,7,6} {2,6,7,3}                 2,3,7,6
  };
  const static int femfaceindex[6] = {1, 0, 4, 2, 3, 5};

  //Res
  const static int resfaces[6][4] = {
    {0,2,3,1}, //0,2,3,1
    {4,5,7,6}, //4,5,7,6
    {1,3,7,5}, //1,3,7,5 
    {0,4,6,2}, //0,4,6,2
    {0,1,5,4}, //0,1,5,4
    {2,6,7,3}  //2,6,7,3
  };
  const static int resfaceindex[6] = {0,1,2,3,4,5};
  
  if(m_resGeometry == NULL){
    faceindex = (int *)femfaceindex;
    for(int i=0; i<6; i++)
      faces[i] = (int *)femfaces[i];
  }    
  else{
    faceindex = (int *)resfaceindex;
    for(int i=0; i<6; i++)
      faces[i] = (int *)resfaces[i];
  }
  
  //Adjacencies and normal calculation
  int ti = 0;
  float sumdistance = 0;
  int coplanarcounter = 0;
  int facecounter = 0;
  int degeneratecounter = 0;
  int fi = 0;
  int textouse[6] = {0, 0, 0, 1, 1, 1};

  //Store the normal used for each face, so the adjacent face uses the same vertices to calculate its normals
  bool* hasnormal = new bool[6 * (m_topmodel->GetNElem() + 1)];
  AlgVector* coplanarpoints1 = new AlgVector[6 * (m_topmodel->GetNElem() + 1)];
  AlgVector* coplanarpoints2 = new AlgVector[6 * (m_topmodel->GetNElem() + 1)];

  memset((bool *)hasnormal, 0, 6 * (m_topmodel->GetNElem() + 1) * sizeof(bool));

  for (TopModel::ElemItr itr(m_topmodel); itr.IsValid(); itr.Next()){
    TopElement el = itr.GetCurr();

    if(el.IsValid() && ((m_resGeometry == NULL) || (m_resGeometry != NULL && m_resGeometry->IsActiveCell(m_topmodel->GetId(el))))){

      //Nodes
#ifdef CUDARC_BILINEAR
      for(int ni=0; ni<8; ni++){
        AlgVector p;
        m_topmodel->GetPosition(m_topmodel->GetNode(el, ni), &(p.x), &(p.y), &(p.z));

        collisionData[ni][4 * (ti+1) + 0] = p.x;
        collisionData[ni][4 * (ti+1) + 1] = p.y;
        collisionData[ni][4 * (ti+1) + 2] = p.z;
        collisionData[ni][4 * (ti+1) + 3] = 1;
      }
#endif

      for(int fi = 0; fi < 6; fi++){

        AlgVector normal1 = AlgVector(0, 0, 0);
        AlgVector normal2 = AlgVector(0, 0, 0);
        AlgVector p[4];
        AlgVector coplanarpoint1;
        AlgVector coplanarpoint2;
        float distance = 0;

        m_topmodel->GetPosition(m_topmodel->GetNode(el, faces[fi][0]), &(p[0].x), &(p[0].y), &(p[0].z));
        m_topmodel->GetPosition(m_topmodel->GetNode(el, faces[fi][1]), &(p[1].x), &(p[1].y), &(p[1].z));
        m_topmodel->GetPosition(m_topmodel->GetNode(el, faces[fi][2]), &(p[2].x), &(p[2].y), &(p[2].z));
        m_topmodel->GetPosition(m_topmodel->GetNode(el, faces[fi][3]), &(p[3].x), &(p[3].y), &(p[3].z));

        TopElement adj = m_topmodel->GetAdj(el, faceindex[fi]);
        if(adj.IsValid() && ((m_resGeometry == NULL) || (m_resGeometry != NULL && m_resGeometry->IsActiveCell(m_topmodel->GetId(adj))))){
          int id = m_topmodel->GetId(adj);
          int adjid = m_elemId[id];
          
          /*
          int adjfaceid = 0;
          for(int i=0; i<6; i++){
            TopElement adjaux = m_topmodel->GetAdj(id, i);

            if(adjaux.IsValid() && adjaux.GetHandle() - 1 == ti){
              adjfaceid = i;
              break;
            }
          }
          */

          //Save both the element id and the face id of the --adjacent-- cell
          static int aux[6] = {1,0,3,2,5,4};
          adjacenciesData[textouse[fi]][4 * (ti+1) + fi - 3 * textouse[fi]] = 6 * adjid + aux[fi];

#ifndef CUDARC_BILINEAR
          //Check if adj has normal.
          if(hasnormal[6*adjid + aux[fi]]){
            

            //Use same normal (with opposite direction)       
            normal1.x = - collisionData[aux[fi]][4 * (adjid) + 0];
            normal1.y = - collisionData[aux[fi]][4 * (adjid) + 1];
            normal1.z = - collisionData[aux[fi]][4 * (adjid) + 2];
            coplanarpoint1 = coplanarpoints1[6 * (adjid) + aux[fi]];
            /*
#ifdef CUDARC_TWONORMALS_PER_PATCH
            normal2.x = - collisionData[aux[fi]][1][4 * (adjid) + 0];
            normal2.y = - collisionData[aux[fi]][1][4 * (adjid) + 1];
            normal2.z = - collisionData[aux[fi]][1][4 * (adjid) + 2];
            coplanarpoint2 = coplanarpoints2[6 * (adjid) + aux[fi]];
#endif
            */

            /*
            AlgVector aux1;
            AlgVector aux2;
            AlgVector auxpoint1;
            AlgVector auxpoint2;
            ComputeHexNormal(p[0], p[1], p[2], p[3], &aux1, &aux2, &auxpoint1, &auxpoint2);

            if(aux1 != normal1 || aux2 != normal2 || auxpoint1 != coplanarpoint1 || auxpoint2 != coplanarpoint2)
              printf("Diff\n");
            else
              printf("Eq.\n");
            */
          }
          else{
            //Calculate a normal
            distance = ComputeHexNormal(p[0], p[1], p[2], p[3], &normal1, &normal2, &coplanarpoint1, &coplanarpoint2);
          }

          coplanarpoints1[6 * (ti+1) + fi] = coplanarpoint1;
          coplanarpoints2[6 * (ti+1) + fi] = coplanarpoint2;
          hasnormal[6*(ti+1) + fi] = true;
          
          
          //Calculate a normal
          distance = ComputeHexNormal(p[0], p[1], p[2], p[3], &normal1, &normal2, &coplanarpoint1, &coplanarpoint2);
          coplanarpoints1[6 * (ti+1) + fi] = coplanarpoint1;
          coplanarpoints2[6 * (ti+1) + fi] = coplanarpoint2;
          hasnormal[6*(ti+1) + fi] = true;
          
        }
        else{
          adjacenciesData[textouse[fi]][4 * (ti+1) + fi - 3 * textouse[fi]] = 0;

          distance = ComputeHexNormal(p[0], p[1], p[2], p[3], &normal1, &normal2, &coplanarpoint1, &coplanarpoint2);
          coplanarpoints1[6 * (ti+1) + fi] = coplanarpoint1;
          coplanarpoints2[6 * (ti+1) + fi] = coplanarpoint2;
          hasnormal[6*(ti+1) + fi] = true;
        }

        if(normal1.Length() > 0){
          facecounter++;
          if(distance == 0)
            coplanarcounter++;
        }
        else
          degeneratecounter++;

        sumdistance += distance;



        collisionData[fi][4 * (ti+1) + 0] = normal1.x;
        collisionData[fi][4 * (ti+1) + 1] = normal1.y;
        collisionData[fi][4 * (ti+1) + 2] = normal1.z;
        collisionData[fi][4 * (ti+1) + 3] = -(coplanarpoint1.Dot(normal1));
        //collisionData[fi][1][4 * (ti+1) + 0] = normal2.x;
        //collisionData[fi][1][4 * (ti+1) + 1] = normal2.y;
        //collisionData[fi][1][4 * (ti+1) + 2] = normal2.z;
        //collisionData[fi][1][4 * (ti+1) + 3] = -(coplanarpoint2.Dot(normal2));
#else
        }
#endif
      }
      ti++;
    }
  }

  delete [] hasnormal;
  delete [] coplanarpoints1;
  delete [] coplanarpoints2;

#ifdef CUDARC_VERBOSE
  printf("\nTotal faces: %d.\n4-point co-planars: %d (%f)\nDegenerate faces: %d\nAvg. distance: %f (Max edge: %f)\n",
          facecounter, coplanarcounter, coplanarcounter/(float)facecounter, degeneratecounter, (double)sumdistance / (double)facecounter,  m_tetraGeometry->GetMaxEdgeLength());
  printf("Done.\n");
#endif
}

MODEL_CLASS_TEMPLATE
float CudaRC<MODELCLASS>::determinant(const float entries[9])
{
	return 	(-entries[6]*entries[4]*entries[2] - entries[7]*entries[5]*entries[0] - entries[8]*entries[3]*entries[1]
	+ entries[0]*entries[4]*entries[8] + entries[1]*entries[5]*entries[6] + entries[2]*entries[3]*entries[7]);
}


MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::BuildTetraVertexGradientTextures(float** gradientVertexData, float** gradientData){

#ifdef CUDARC_VERBOSE
  printf("\nBuilding Tetra Vertex Gradient Textures... ");
#endif

  TpvTetrahedronSet* set = m_tetraGeometry->GetTetrahedra();
  TpvIntArray *tetraIndex = set->GetVertexIncidences();
  TpvFloatArray* vertexPositions = set->GetVertexPositions();

  int aux = - INT_MAX;
  for( int i = 0; i < tetraIndex->Size(); i++ )
  {
	if( aux < (*tetraIndex)[i] )
		aux = (*tetraIndex)[i];
  }
  int nVertex = aux;
  std::vector<float> vertexGradientAux;
  vertexGradientAux.resize(4*nVertex);

  float fator = 1.0/6.0;
  std::vector<float> gradientVolAcummulation;
  gradientVolAcummulation.resize(nVertex);

  float maior = -FLT_MAX;

  for( int i = 0 ; i < m_memoryInfo.numElem; i++ ){
		//Quarto vertices do tetraedro
		int idV1 = (*tetraIndex)[4*i] - 1;
		int idV2 = (*tetraIndex)[4*i+1] - 1;
		int idV3 = (*tetraIndex)[4*i+2] - 1;
		int idV4 = (*tetraIndex)[4*i+3] - 1;

			//vertices do primeiro tetraedro
		//VECTOR3D v1((*tet_vextex_positions)[3*idVertex1],(*tet_vextex_positions)[3*idVertex1+1],(*tet_vextex_positions)[3*idVertex1+2]);
		//VECTOR3D v2((*tet_vextex_positions)[3*idVertex2],(*tet_vextex_positions)[3*idVertex2+1],(*tet_vextex_positions)[3*idVertex2+2]);
		//VECTOR3D v3((*tet_vextex_positions)[3*idVertex3],(*tet_vextex_positions)[3*idVertex3+1],(*tet_vextex_positions)[3*idVertex3+2]);
		//VECTOR3D v4((*tet_vextex_positions)[3*idVertex4],(*tet_vextex_positions)[3*idVertex4+1],(*tet_vextex_positions)[3*idVertex4+2]);

		AlgVector v1((*vertexPositions)[3*idV1],(*vertexPositions)[3*idV1+1],(*vertexPositions)[3*idV1+2]);
		AlgVector v2((*vertexPositions)[3*idV2],(*vertexPositions)[3*idV2+1],(*vertexPositions)[3*idV2+2]);
		AlgVector v3((*vertexPositions)[3*idV3],(*vertexPositions)[3*idV3+1],(*vertexPositions)[3*idV3+2]);
		AlgVector v4((*vertexPositions)[3*idV4],(*vertexPositions)[3*idV4+1],(*vertexPositions)[3*idV4+2]);

		AlgVector u = v1 - v4;
		AlgVector v = v2 - v4;
		AlgVector w = v3 - v4;

		float entries[9] = {u.x, u.y, u.z, v.x, v.y, v.z, w.x, w.y, w.z};

		float volume = abs(fator*determinant( entries ));
		//float volume = fator*abs(produtoMisto.determinant());

		if( maior < volume )
			maior = volume;

		vertexGradientAux[4*idV1] += gradientData[0][4*(i+1)]*volume;
		vertexGradientAux[4*idV1+1] += gradientData[0][4*(i+1)+1]*volume;
		vertexGradientAux[4*idV1+2] += gradientData[0][4*(i+1)+2]*volume;
		vertexGradientAux[4*idV1+3] += gradientData[0][4*(i+1)+3]*volume;

		vertexGradientAux[4*idV2] += gradientData[0][4*(i+1)]*volume;
		vertexGradientAux[4*idV2+1] += gradientData[0][4*(i+1)+1]*volume;
		vertexGradientAux[4*idV2+2] += gradientData[0][4*(i+1)+2]*volume;
		vertexGradientAux[4*idV2+3] += gradientData[0][4*(i+1)+3]*volume;

		vertexGradientAux[4*idV3] += gradientData[0][4*(i+1)]*volume;
		vertexGradientAux[4*idV3+1] += gradientData[0][4*(i+1)+1]*volume;
		vertexGradientAux[4*idV3+2] += gradientData[0][4*(i+1)+2]*volume;
		vertexGradientAux[4*idV3+3] += gradientData[0][4*(i+1)+3]*volume;

		vertexGradientAux[4*idV4] += gradientData[0][4*(i+1)]*volume;
		vertexGradientAux[4*idV4+1] += gradientData[0][4*(i+1)+1]*volume;
		vertexGradientAux[4*idV4+2] += gradientData[0][4*(i+1)+2]*volume;
		vertexGradientAux[4*idV4+3] += gradientData[0][4*(i+1)+3]*volume;

		gradientVolAcummulation[idV1]+=volume;
		gradientVolAcummulation[idV2]+=volume;
		gradientVolAcummulation[idV3]+=volume;
		gradientVolAcummulation[idV4]+=volume;
  }

  	for(int i = 0; i < nVertex; i++ )
	{
		vertexGradientAux[4*i] /= gradientVolAcummulation[i];
		vertexGradientAux[4*i+1] /= gradientVolAcummulation[i];
		vertexGradientAux[4*i+2] /= gradientVolAcummulation[i];
		vertexGradientAux[4*i+3] /= gradientVolAcummulation[i];
	}


  for( int i = 0 ; i < m_memoryInfo.numElem; i++ ){
		//Quarto vertices do tetraedro
	    int id[4];
		id[0] = (*tetraIndex)[4*i] - 1;
		id[1] = (*tetraIndex)[4*i+1] - 1;
		id[2] = (*tetraIndex)[4*i+2] - 1;
		id[3] = (*tetraIndex)[4*i+3] - 1;
		
		for(int j = 0; j < 4; j++ ){
			gradientVertexData[j][4 * (i+1) + 0] = vertexGradientAux[4*id[j] + 0];
			gradientVertexData[j][4 * (i+1) + 1] = vertexGradientAux[4*id[j] + 1];
			gradientVertexData[j][4 * (i+1) + 2] = vertexGradientAux[4*id[j] + 2];
			gradientVertexData[j][4 * (i+1) + 3] = vertexGradientAux[4*id[j] + 3];
		}
  }

#ifdef CUDARC_VERBOSE
  printf("Done.\n");
#endif
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::BuildTetraMeshTextures(float** collisionData, float** adjacenciesData){
#ifdef CUDARC_VERBOSE
  printf("\nBuilding Mesh Textures... ");
#endif

  TpvTetrahedronSet* set = m_tetraGeometry->GetTetrahedra();
  int numFaces = 4;
  static const int F_V0[4] = {1, 2, 3, 0};
  
#ifdef CUDARC_PLUCKER
  //Nodes
  for(int ti=0; ti<m_memoryInfo.numElem; ti++){
    for(int ni=0; ni<4; ni++){
      float* pos = set->GetPosition(ti, ni);
      collisionData[ni][4 * (ti+1) + 0] = pos[0];
      collisionData[ni][4 * (ti+1) + 1] = pos[1];
      collisionData[ni][4 * (ti+1) + 2] = pos[2];
      collisionData[ni][4 * (ti+1) + 3] = 1;
    }
  }
#else
  //Faces planes equations
  for(int fi=0; fi<numFaces; fi++){
    for(int ti=0; ti<m_memoryInfo.numElem; ti++){
      float* normal = set->GetNormal(ti, fi);
      float* pos = set->GetPosition(ti, F_V0[fi]);
 
      collisionData[fi][4 * (ti+1) + 0] = normal[0];
      collisionData[fi][4 * (ti+1) + 1] = normal[1];
      collisionData[fi][4 * (ti+1) + 2] = normal[2];
      collisionData[fi][4 * (ti+1) + 3] = -(pos[0]*normal[0] + pos[1]*normal[1] + pos[2]*normal[2]);
    }
  }
#endif

  

  //Adjacencies
  for(int fi=0; fi<numFaces; fi++){
    for(int ti=0; ti<m_memoryInfo.numElem; ti++){
      int posindices[3] = {-1, -1, -1};
       int adjid = set->GetAdj(ti, fi); // 1-indexed
      float* adjpos[4];
      float* tpos[4];
      int adjfaceid = 0;

      for(int i=0; i<4; i++){
        adjpos[i] = set->GetPosition(adjid-1, i);
        tpos[i] = set->GetPosition(ti, i);
      }

      if(adjid > 0){
        for(int i=0; i<4; i++){
          int adjaux = set->GetAdj(adjid - 1, i);
          
          if(adjaux - 1 == ti){
            adjfaceid = i;
            break;
          }
        }
      }

      //Save both the element id and the face id of the --adjacent-- cell
      static int aux[4] = {0, 1, 2, 3};
      adjacenciesData[0][numFaces * (ti+1) + fi] = 4 * adjid + aux[adjfaceid]; //<<---- aqui
      //adjacenciesData[0][numFaces * (ti+1) + fi] = id;
      
    }
  }


#ifdef CUDARC_VERBOSE
  printf("Done.\n");
#endif
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::BuildHexaInterpolFuncTexture(float** gradientData){
#ifdef CUDARC_VERBOSE
  printf("\nBuilding Interpolation Function Textures... ");
#endif

  //static const int vertexAdj[8][3] = {{1, 2, 4}, {0, 3, 5}, {0, 3, 6}, {1, 2, 7}, {0, 5, 6}, {1, 4, 7}, {2, 4, 7}, {3, 5, 6}};
  //static const int elemAdj[8][3] = {{0, 3, 4}, {0, 2, 4}, {0, 2, 5}, {0, 1, 2}, {1, 3, 4}, {1, 2, 4}, {1, 3, 5}, {1, 2, 5}};

  int ti = 0;
  for(TopModel::ElemItr itr(m_topmodel); itr.IsValid(); itr.Next()){
    TopElement el = itr.GetCurr();
    int id = m_topmodel->GetId(el);

    if((m_resGeometry == NULL) || (m_resGeometry != NULL && m_resGeometry->IsActiveCell(id))){

      float gradient[8];
      float scalars[8];
      float position[24];
      TopNode nodes[8];

      for(int j=0; j<8; j++){
        nodes[j] = m_topmodel->GetNode(el, j);
        m_topmodel->GetPosition(nodes[j], &(position[3*j]), &(position[3*j+1]), &(position[3*j+2]));
        float* aux = m_property->GetValue(m_topmodel->GetId(nodes[j]));
        scalars[j] = *aux;
        if(m_property->GetMaxValue()[0] == m_property->GetMinValue()[0])
          scalars[j] = *aux;//m_property->GetMaxValue()[0];
        else
          scalars[j] = m_property->NormalizeScalar(*aux);
      }

      ComputeGradientLeastSquares(8, position, scalars, gradient);

      //Test for NaN
      assert(gradient[0] == gradient[0]);


      gradientData[0][4 * (ti+1)] = gradient[0];
      gradientData[0][4 * (ti+1) + 1] = gradient[1];
      gradientData[0][4 * (ti+1) + 2] = gradient[2];
      gradientData[0][4 * (ti+1) + 3] = gradient[3];
      gradientData[1][4 * (ti+1)] = gradient[4];
      gradientData[1][4 * (ti+1) + 1] = gradient[5];
      gradientData[1][4 * (ti+1) + 2] = gradient[6];
      gradientData[1][4 * (ti+1) + 3] = gradient[7];
    
      ti++;
    }
    
    
  }
#ifdef CUDARC_VERBOSE
  printf("Done.\n");
#endif
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::BuildTetraInterpolFuncTexture(float** gradientData){
#ifdef CUDARC_VERBOSE
  printf("\nBuilding Interpolation Function Textures... ");
#endif

  TpvTetrahedronSet* set = m_tetraGeometry->GetTetrahedra();
  float gradient[4];
  
  TpvFloatArray* arrayPositions = set->GetPositions();
  TpvFloatArray* arrayScalars = set->GetField();

  for(int ti=0; ti<m_memoryInfo.numElem; ti++){
    
    float* position = &(arrayPositions->GetArray()[4 * 3 * ti]);
    float* scalars = &(arrayScalars->GetArray()[4 * ti]);

    
    ComputeGradientLeastSquares(4, position, scalars, gradient);
    
    
    //float gradient2[3];
    //ComputeGradientLeastSquares(4, 0, position, scalars, gradient);
    //ComputeGradient(position, scalars, gradient2);
    /*
    for(int i=0; i<3; i++){
      if(fabs(gradient[i] - gradient2[i]) > 1e-1){
        //gradient[0] = gradient2[0];
        //gradient[1] = gradient2[1];
        //gradient[2] = gradient2[2];
      }
    }
    */
    //Test for NaN
    assert(gradient[0] == gradient[0]);
    
    
    gradientData[0][4 * (ti+1)] = gradient[0];
    gradientData[0][4 * (ti+1) + 1] = gradient[1];
    gradientData[0][4 * (ti+1) + 2] = gradient[2];
    //scalar term, for linear tetrahedral meshes
    //gradientData[0][4 * (ti+1) + 3] = scalars[0] - (gradient[0]*position[0] + gradient[1]*position[1] + gradient[2]*position[2]);
    gradientData[0][4 * (ti+1) + 3] = gradient[3];
    /*
    gradientData[0][4 * (ti+1)] = scalars[0];
    gradientData[0][4 * (ti+1) + 1] = scalars[1];
    gradientData[0][4 * (ti+1) + 2] = scalars[2];
    gradientData[0][4 * (ti+1) + 3] = scalars[3];
    */
  }
#ifdef CUDARC_VERBOSE
  printf("Done.\n");
#endif
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::BuildAmbOcclusionTexture(float** ambocclusdata){
#ifdef CUDARC_VERBOSE
  printf("\nBuilding Interpolation Function Textures... ");
#endif
/*
  TpvTetrahedronSet* set = m_tetraGeometry->GetTetrahedra();
  float gradient[4];
  
  TpvFloatArray* arrayPositions = set->GetPositions();
  TpvFloatArray* arrayScalars = set->GetField();

  for(int ti=0; ti<m_memoryInfo.numElem; ti++){
    
    float* position = &(arrayPositions->GetArray()[4 * 3 * ti]);
    float* scalars = &(arrayScalars->GetArray()[4 * ti]);

    
    ComputeGradientLeastSquares(4, position, scalars, gradient);
    
    gradientData[0][4 * (ti+1)] = gradient[0];
    gradientData[0][4 * (ti+1) + 1] = gradient[1];
    gradientData[0][4 * (ti+1) + 2] = gradient[2];
    gradientData[0][4 * (ti+1) + 3] = gradient[3];

  }
  */
#ifdef CUDARC_VERBOSE
  printf("Done.\n");
#endif
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::BuildColorScaleTexture(float* volcolorscaledata, float* isocolorscaledata){
#ifdef CUDARC_VERBOSE
  printf("\nBuilding Color Scale Textures... ");
#endif

  m_volcolorscale->GetTexArray(m_memoryInfo.numValuesTf, volcolorscaledata);
  m_isocolorscale->GetTexArray(m_memoryInfo.numValuesTf, isocolorscaledata);

#ifdef CUDARC_VERBOSE
  printf("Done.\n");
#endif
}


/*
1D texture contained the isosurfaces
Size: (transfer function size) * float4
For position n:
- n (index): scalar
- n.x : first control point (i) larger than n
- n.y: next control point (i+1)

- n.z: first control point (j) smaller than n
- n.w: previous control point (j-1)
*/
MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::BuildControlPointsTexture(float* cpdata, float numcp, float* cpvalues, float smin, float smax){
#ifdef CUDARC_VERBOSE
  printf("\nBuilding Control Points Texture... ");
#endif

  float sdiff = smax - smin;

  //First control point (cp) bigger than s
  int cpCounter = 0;
  for(int i=0; i< m_memoryInfo.numValuesTf; ){
    float s = (float) (i) / (float) (m_memoryInfo.numValuesTf);
    float s_cp = (cpvalues[cpCounter] - smin) / (sdiff);

    if(cpCounter < numcp - 1){
      if(s <= s_cp){
        cpdata[4 * i] = s_cp;
        cpdata[4 * i + 1] = (cpvalues[cpCounter+1] - smin) / (sdiff);
        i++;
      }
      else{
        cpCounter ++;
      }
    }
    else{ /* Last cp */
      if(s <= s_cp){
        cpdata[4 * i] = s_cp;
        cpdata[4 * i + 1] = 3.0f;
      }
      i++;
    }

  }

  //First control point (cp) smaller than s
  for(int i=m_memoryInfo.numValuesTf - 1; i>= 0 ; ){
    float s = (float) (i+1.0f) / (float) (m_memoryInfo.numValuesTf-1.0f);
    float s_cp = (cpvalues[cpCounter] - smin) / (sdiff);

    if(cpCounter > 0){
      if(s >= s_cp){
        cpdata[4 * i + 2] = s_cp;
        cpdata[4 * i + 3] = (cpvalues[cpCounter-1] - smin) / (sdiff);
        i--;
      }
      else{
        cpCounter --;
      }
    }
    else{
      if(s >= s_cp){
        cpdata[4 * i + 2] = s_cp;
        cpdata[4 * i + 3] = 3.0f;
      }
      i--;
    }
  }
#ifdef CUDARC_VERBOSE
  printf("Done.\n");
#endif
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::Render(bool bdryonly, float* eyepos, float* eyeDir, float* eyeUp, float eyeZNear, float eyeFov, bool debug, float delta, float deltaW, float zero){

  
  Update();


  //Depth peeling pass 0 to 255
  int currentIndex = 0;
  int backIndex = 1;
  float kernelCallTime = 0;
  float overheadTime = 0;
  double shaderTime = 0;

#ifdef CUDARC_TIME
  m_time.Reset();  
#endif

  int maxnumpeel = m_maxNumPeel;
  
  if(bdryonly){
    if(m_numpeeling == 0) maxnumpeel = 1;
    else maxnumpeel = m_numpeeling;
  }

  for(int i=0; i<maxnumpeel; i++){

#ifdef CUDARC_TIME
    glFinish();
    shaderTime = uso_gettime();
    //int handleTime = 0;
    //GLuint shaderTime = 0;
    //glBeginQuery(GL_TIME_ELAPSED, handleTime);
    kernelCallTime = 0;
    overheadTime = 0;
#endif

    int occQueryCount = 0;
    
    
    if(bdryonly == false) m_uglFbIntersect[currentIndex].Activate(true);
    {
      m_shaderIntersect->Load();
      m_uglDepthIntersect[backIndex].Load();
      {
        m_shaderIntersect->SetConstant(Shader::FP, "eyePos", eyepos[0], eyepos[1], eyepos[2]);
        m_shaderIntersect->SetConstant(Shader::FP, "depthPeelPass", (float)i);
        m_shaderIntersect->SetConstant(Shader::FP, "winSize", m_winsize.x, m_winsize.y);
        m_shaderIntersect->BindTexture("depthSampler", m_uglDepthIntersect[backIndex].GetLoadedUnit());

        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_EQUAL, 1, 1);
        glStencilOp(GL_ZERO, GL_KEEP, GL_KEEP);
        glBeginQuery(GL_SAMPLES_PASSED, m_handleOccQuery);
        RenderSinglePass(eyepos);
        glEndQuery(GL_SAMPLES_PASSED);
        glGetQueryObjectiv(m_handleOccQuery, GL_QUERY_RESULT, &occQueryCount);
      }
      m_uglDepthIntersect[backIndex].Unload();
      m_shaderIntersect->Unload();
    }
    if(bdryonly == false) m_uglFbIntersect[currentIndex].Deactivate(false, false);
    
    if(occQueryCount == 0)
      break;

#ifdef CUDARC_TIME
    glFinish();
    shaderTime = uso_gettime() - shaderTime;
    //glEndQuery(GL_TIME_ELAPSED);
    //glGetQueryObjectuiv(handleTime, GL_QUERY_RESULT, &shaderTime);
#endif
    

    //CUDA
    if(bdryonly == false){
      if(m_numpeeling == 0)
        run(&kernelCallTime, &overheadTime, i, eyepos, (float*)m_probeboxmin, (float*)m_probeboxmax, m_uglTexIntersect[currentIndex].GetTextureId(), delta, deltaW, zero);
      else if(m_numpeeling == i)
        run(&kernelCallTime, &overheadTime, 0, eyepos, (float*)m_probeboxmin, (float*)m_probeboxmax, m_uglTexIntersect[currentIndex].GetTextureId(), delta, deltaW, zero);
    }

    //Swap
    currentIndex = currentIndex == 0 ? 1 : 0;
    backIndex = backIndex == 0 ? 1 : 0;

#ifdef CUDARC_TIME
    m_time.numPeel = i+1;
    m_time.SetTime(i, shaderTime * 1000.0, kernelCallTime, overheadTime, occQueryCount);
#endif

  }

#ifdef CUDARC_TIME
  m_time.currentFrame++;
#endif

}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::RenderSinglePass(float* eyePos){

  glClearColor(1.0, 1.0, 1.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);  
  //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  glFrontFace(GL_CCW);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glCallList(m_bdryDispList[0]);
  glPopAttrib();
  

}


MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::DisplayInfo(double totaltime){

  //TODO: replace with UFontManager
  UFontGLUTMessage message;
  char str[1000];
  int size = 0;
  double totalcudarctime = (m_time.lastShaderTime + m_time.lastKcTime + m_time.lastOvTime);// / m_time.numFrames;

  size += sprintf(str+size, "Num peel.: %03d, ", m_time.numPeel);
  size += sprintf(str+size, "S: %07.3f, ", m_time.lastShaderTime);// / m_time.numFrames);
  size += sprintf(str+size, "KC: %07.3f, ", m_time.lastKcTime);//  / m_time.numFrames);
  size += sprintf(str+size, "OV: %07.3f, ", m_time.lastOvTime);//  / m_time.numFrames);

  size += sprintf(str+size, "T: %07.3f", totalcudarctime);

#ifdef CUDARC_WHITE
  message.SetColor(0, 0, 0);
#else
  message.SetColor(1, 1, 1);
#endif
  message.Display(str);
  
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::PrintInfo(){

  printf("#Arguments: \n");
  printf("#maxedge: %d, blocksize: %d, maxpeel: %d, tfsize: %d, win: %d x %d\n", m_maxedgeenabled, m_blocksize.x, m_maxNumPeel, m_memoryInfo.sizeTf, (int)m_winsize.x, (int)m_winsize.y);

  printf("# - Num elements: %d\n", m_memoryInfo.numElem);
  unsigned int totalMemory = m_memoryInfo.numAdjTex * m_memoryInfo.sizeAdjTex
                            + m_memoryInfo.numNodes * m_memoryInfo.sizeCollisionTex
                            + m_memoryInfo.numInterpolFuncTex * m_memoryInfo.sizeInterpolFuncTex;
  printf("# - Memory usage: \n");
  printf("#       Elements adj.: %u (T: %d)\n#       Collision.: %u (T: %u)\n#       Interpol Func: %u (T: %u)\n#       Total per elem: %u (T: %u)\n",
          (m_memoryInfo.numAdjTex * m_memoryInfo.sizeAdjTex) / m_memoryInfo.numElem,
          m_memoryInfo.numAdjTex * m_memoryInfo.sizeAdjTex,
          (m_memoryInfo.numNodes * m_memoryInfo.sizeCollisionTex) / m_memoryInfo.numElem,
          m_memoryInfo.numNodes * m_memoryInfo.sizeCollisionTex,
          (m_memoryInfo.numInterpolFuncTex * m_memoryInfo.sizeInterpolFuncTex) / m_memoryInfo.numElem,
          m_memoryInfo.numInterpolFuncTex * m_memoryInfo.sizeInterpolFuncTex,
          totalMemory / m_memoryInfo.numElem,
          totalMemory);
#ifdef CUDARC_HEX
  printf("#       ZetaPsiGamma: %u (%ux%ux%ux4)\n", m_memoryInfo.sizeZetaPsiGamma, m_memoryInfo.numValuesZetaPsiGamma, m_memoryInfo.numValuesZetaPsiGamma, m_memoryInfo.numValuesZetaPsiGamma);
#else
  printf("#       ZetaPsiGamma: %u (%ux%ux1)\n", m_memoryInfo.sizeZetaPsiGamma, m_memoryInfo.numValuesZetaPsiGamma, m_memoryInfo.numValuesZetaPsiGamma);
#endif

  printInfoGPUMemory();

}


MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::PrintTime(){

  printf("===\nTime results\n");

  double totalTime = (m_time.totalShaderTime + m_time.totalKcTime + m_time.totalOvTime) / m_time.numFrames;
  printf(" - Total time (avg. %03d frames):\n       S: %f, KC: %f, OV: %f\n       S: (%03.1f%%) KC: (%03.1f%%) OV: (%03.1f%%)\n       Sum: %f\n",
          m_time.currentFrame,
          m_time.lastShaderTime/m_time.numFrames,
          m_time.lastKcTime/m_time.numFrames,
          m_time.lastOvTime/m_time.numFrames,
          100.0 * (m_time.lastShaderTime / m_time.numFrames) / totalTime,
          100.0 * (m_time.lastKcTime / m_time.numFrames) / totalTime,
          100.0 * (m_time.lastOvTime / m_time.numFrames) / totalTime,
          totalTime);
  printf("===\n");
}

MODEL_CLASS_TEMPLATE
GLuint CudaRC<MODELCLASS>::GetPboOutputId(){

	return m_handlePboOutput;
  //return m_uglDepthIntersect.GetTextureId();
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetModel(TopModel* topmodel){

  m_topmodel = topmodel;

}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetModel(TopMultiModel* topmodel){

  SetModel(topmodel->GetModel(0));

}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetGeometry(TpvTetraGeometry3<MODELCLASS>* tetraGeometry){

  m_tetraGeometry = tetraGeometry;
  m_update_geometry = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetResGeometry(ResGeometry* resGeometry){

  m_resGeometry = resGeometry;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetVolumetricColorScale(TpvColorScale* colorScale){

  m_volcolorscale = colorScale;
  m_update_colorscale = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetIsoColorScale(TpvColorScale* colorScale, float* isovalues){

  m_isocolorscale = colorScale;
  m_isovalues = isovalues;
  m_update_colorscale = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetProperty(TpvProperty* prop){

  m_property = prop;
  m_update_property = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetInterpolationType(int interpoltype){
  m_interpoltype = interpoltype;
  m_update_cuda = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetNumSteps(int numsteps){
  m_numsteps = numsteps;
  m_update_cuda = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetNumTraverses(int numtraverses){
  m_numtraverses = numtraverses;
  m_update_cuda = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetNumPeeling(int numpeel){
  m_numpeeling = numpeel;
  //m_update_cuda = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetMaxEdgeLengthEnabled(bool flag){
  m_maxedgeenabled = flag;
  m_update_cuda = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetDebugEnabled(bool flag){
  m_debug = flag;
  m_update_cuda = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetIsoSurfaceEnabled(bool flag){
  m_isosurface = flag;
  m_update_cuda = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetVolumetricEnabled(bool flag){
  m_volumetric = flag;
  m_update_cuda = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetProbeBoxEnabled(bool flag){
  m_probeboxenabled = flag;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetNormalizedField(bool flag){
  m_normalizedfield = flag;
  m_update_geometry = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetProbeBox(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax){
  m_probeboxmin.x = xmin;
  m_probeboxmin.y = ymin;
  m_probeboxmin.z = zmin;
  m_probeboxmax.x = xmax;
  m_probeboxmax.y = ymax;
  m_probeboxmax.z = zmax;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetExplodedView(bool flag){
  m_explodedview = flag;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetBlockSize(int sizex, int sizey){
  m_blocksize = AlgVector(sizex, sizey, 0);
  m_update_cuda = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetWindowSize(int sizex, int sizey){
  m_winsize = AlgVector(sizex, sizey, 0);
  m_update_window = true;
  m_update_cuda = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetMaxNumPeel(int maxpeel){
  m_maxNumPeel = maxpeel;
  m_update_maxnumpeel = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetTessellation(int tessellation){
  m_tessellation = tessellation;
  m_update_geometry = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetPaths(int zetapsigammasize, const char* zetapsigammapath, const char* shaderpath){
  m_zetapsigammasize = zetapsigammasize;
  m_zetapsigammapath = zetapsigammapath;
  m_shaderpath = shaderpath;
  m_update_zetapsigamma = true;
}

MODEL_CLASS_TEMPLATE
void CudaRC<MODELCLASS>::SetExplosionFactor(float explosionfactor){

  m_explosionfactor = explosionfactor;

}