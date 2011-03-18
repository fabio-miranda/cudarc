/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* t.cudarc.cpp: simple renderer that creates the CudaRC object and handles user interaction
*
* 
* usage: %s modelpath [-propfile] [-normalizefield] [-shaderpath] [-blocksize] [-maxpeel] [-tfpath] [-tfsize] [-winx] [-winy] [-scalez] [-transp] [-outputpath] [-benchmark] [-maxedge] [-interpol] [-tessellation]
* default values:
* modelpath: (non-optional)
* shaderpath: ../../../src/cudarc/glsl/
* blocksize: 32
* maxpeel: 256
* tfpath: ../../../src/cudarc/tf/1.tf
* tfsize: 1024
* zetapsigammapath: ../../../src/cudarc/zetapsigamma/
* zetapsigammasize: 512
* winx: 512
* winy: 512
* scalez: 10
* maxedge: 1
* normalizefield: 1
* interpol: const, linear, trilinear, quad, step (default: step)
* numsteps: 10 (only valid if interpol is step)
* transp: 1
* outputpath: NULL
* bdryonly: 0
* benchmark
* tessellation: 1
*/


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <direct.h>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <vgl/vglglut.h>
#include <gpos/io/open.h>
#include <gpos/model/msh/meshio.h>
#include <gpos/model/pposp/pposp.h>
#include <gpos/model/cmg/cmg.h>
#include <gpos/model/ecl/eclipse.h>
#include <gpos/model/model.h>
#include <gpos/model/geometry.h>
#include <gpos/model/modelnew.h>
#include <tops/readers/plot3d_reader.h>
#include <tops/readers/model_reader.h>
#include <tops/util/topsutil.h>
#include <topsview/renderer/harcvolrenderer3.h>
#include <topsview/geometry/restetrageometry3.h>
#include <topsview/geometry/femtetrageometry3.h>
#include <topsview/geometry/quadset.h>
#include <ugl/uglim.h>

#include <cudarc.h>
#include "defines.h"
#include "transferfunc.h"

static TpvFemTetraGeometry3* s_tpvFemTetraGeometry = NULL;
static TpvResTetraGeometry3* s_tpvResTetraGeometry = NULL;
static TopModel* s_topmodel = NULL;
static TopMultiModel* s_topmultimodel = NULL;
static ResGeometry* s_resGeometry = NULL;
static ResProperty* s_resproperty = NULL;
static TpvProperty* s_tpvproperty = NULL;
static TpvColorScale* s_volcolorscale = NULL;
static TpvColorScale* s_isocolorscale = NULL;
static float s_scalarmin = 1.0f;
static float s_scalarmax = 0.0f;
static float* s_isovalues = NULL;
static VglCanvas* s_canvas = NULL;
static VglCamera* s_camera = NULL;
static VglManipHandler* s_handler = NULL;
#ifndef CUDARC_HARC
static CudaRC<TpvFemGeometryModel>* s_femCudaRC = NULL;
static CudaRC<TpvResGeometryModel>* s_resCudaRC = NULL;
#else
static TpvHARCVolRenderer3<TpvFemGeometryModel>* s_femHarc = NULL;
static TpvHARCVolRenderer3<TpvResGeometryModel>* s_resHarc = NULL;
#endif
static float s_eyePos[3];
static float s_eyeDir[3];
static float s_eyeUp[3];
static float s_eyeRight[3];
static float s_eyeZNear;
static float s_eyeZFar;
static float s_eyeFov;
static AlgVector s_bbMin;
static AlgVector s_bbMax;
enum GeoType {Res = 0, Fem = 1};
int currentGeoType;
enum InterpolType {Const = 0, Linear = 1, Quad = 2, Step = 3};
int currentInterpolType = 1;
int currentNumSteps = 10;
int currentNumTraverses = 0;
int currentNumPeeling = 0;
float currentDelta = 0.0f;
float currentDeltaW = 0.0f;
float currentZero = 0.0f;

//Command line arguments
static float s_scalez = 1.0f;
static bool s_transp = true;
static bool s_bdryonly = false;
static bool s_debug = false;
static bool s_probebox = false;
static bool s_displaytime = false;
static bool s_benchmark = false;
static bool s_maxedge = true;
static bool s_isosurface = false;
static bool s_volumetric = true;
static bool s_normalizefield = true;
static int s_averageframes = 1;
static int s_tfsize = 1024;
static int s_zetapsigammasize = 512;
static int s_maxpeel = 256;
static int s_blocksize = 16;
static int s_winx = 512;
static int s_winy = 512;
static int s_tessellation = 1;
const char* s_shaderpath = "../../../src/cudarc/glsl/";
const char* s_zetapsigammapath = "../../../src/cudarc/zetapsigamma/";
const char* s_tfpath = "../../../src/cudarc/tf/1.tf";
const char* s_isopath = "../../../src/cudarc/tf/1.iso";
char* s_modelpath = NULL;
char* s_propfile = NULL;
char* s_outputpath = NULL;
char* s_outputfile = NULL;
int s_outputpathsize = 0;
double s_lastframetime = 0;

static void Benchmark();

/**
* Redraw callback
*/
static void redraw(void* data){

#ifdef CUDARC_TIME
  double auxtime;
  glFinish();
  auxtime = uso_gettime();

  if(s_benchmark){
    s_benchmark = false;
    Benchmark();
  }
#endif

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  s_camera->GetUpdatedPosition(&s_eyePos[0], &s_eyePos[1], &s_eyePos[2]);
  s_camera->GetUpdatedViewDir(&s_eyeDir[0], &s_eyeDir[1], &s_eyeDir[2]);
  s_camera->GetUpdatedUp(&s_eyeUp[0], &s_eyeUp[1], &s_eyeUp[2]);
  s_eyeZNear = s_camera->GetUpdatedZNear();
  s_eyeZFar = s_camera->GetUpdatedZFar();
  s_eyeFov = s_camera->GetUpdatedAngle();

  glPushMatrix();
  glTranslatef(
    (s_bbMin.x+s_bbMax.x)*0.5,
    (s_bbMin.y+s_bbMax.y)*0.5,
    (s_bbMin.z+s_bbMax.z)*0.5);
  glScalef(1, 1, s_scalez);
  glTranslatef(
    -(s_bbMin.x+s_bbMax.x)*0.5,
    -(s_bbMin.y+s_bbMax.y)*0.5,
    -(s_bbMin.z+s_bbMax.z)*0.5);

  AlgVector eye = VglFrustum().GetEyePos();
  s_eyePos[0] = eye.x; s_eyePos[1] = eye.y; s_eyePos[2] = eye.z;

#ifndef CUDARC_HARC
  if(s_bdryonly){
    if(currentGeoType == Fem){
      s_femCudaRC->Render(s_bdryonly, s_eyePos, s_eyeDir, s_eyeUp, s_eyeZNear, s_eyeFov, s_debug, currentDelta, currentDeltaW, currentZero);
      glPopMatrix();
    }
    if(currentGeoType == Res){
      s_resCudaRC->Render(s_bdryonly, s_eyePos, s_eyeDir, s_eyeUp, s_eyeZNear, s_eyeFov, s_debug, currentDelta, currentDeltaW, currentZero);
      glPopMatrix();
    }
  }
  else{
    if(currentGeoType == Fem){
      s_femCudaRC->Render(s_bdryonly, s_eyePos, s_eyeDir, s_eyeUp, s_eyeZNear, s_eyeFov, s_debug, currentDelta, currentDeltaW, currentZero);
      glPopMatrix();
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, s_femCudaRC->GetPboOutputId());
    }
    if(currentGeoType == Res){
      s_resCudaRC->Render(s_bdryonly, s_eyePos, s_eyeDir, s_eyeUp, s_eyeZNear, s_eyeFov, s_debug, currentDelta, currentDeltaW, currentZero);
      glPopMatrix();
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, s_resCudaRC->GetPboOutputId());
    }

    glClearColor( 0.0, 0.0, 0.0, 0.0 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glDrawPixels( s_winx, s_winy, GL_RGBA, GL_FLOAT, 0 );
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  }

#else

  int vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  if(currentGeoType == Res){
    s_resHarc->SetViewport(vp[0], vp[1], vp[2], vp[3]);
    s_resHarc->Render();
  }
  else if(currentGeoType == Fem){
    s_femHarc->SetViewport(vp[0], vp[1], vp[2], vp[3]);
    s_femHarc->Render();
  }
#endif


#ifdef CUDARC_TIME
  glFinish();
  auxtime = uso_gettime() - auxtime;
  s_lastframetime = auxtime * 1000.0;

#ifndef CUDARC_HARC
  if(s_displaytime){
    if(currentGeoType == Fem){
      s_femCudaRC->DisplayInfo(s_lastframetime);
    }
    if(currentGeoType == Res){
      s_resCudaRC->DisplayInfo(s_lastframetime);
    }
  }
  glutSwapBuffers();
#endif
#endif
  

}

/**
* Run benchmark
*/
static void Benchmark(){
  float angle = 5;
  char outputpathtxt[256];
  strcpy(outputpathtxt, s_outputpath);
  sprintf(outputpathtxt+s_outputpathsize, "\\benchmark.txt");

  printf("Starting benchmark...\n");
  printf("Output directory: %s\n", s_outputpath);
  _mkdir(s_outputpath);

  freopen(outputpathtxt,"w",stdout);

#ifndef CUDARC_HARC
  if(currentGeoType == Fem) s_femCudaRC->PrintInfo();
  else if(currentGeoType == Res) s_resCudaRC->PrintInfo();
#endif

  printf("#Total times:\n");
  int pngname = 1;
  double sumframetime = 0;
  int numframes = 0;
  for(int i=0; i<360; i+=angle){
    for(int j=0; j<360; j+=angle){
      s_camera->Rotate(angle, 0, 1, 0);
      s_camera->CenterView();
      s_canvas->Redraw();
      printf("%f\n", s_lastframetime);
      sumframetime += s_lastframetime;
      numframes++;
/*
      if(currentGeoType == Fem) printf("%f\n", s_femCudaRC->m_time.totalKcTime);
      else if(currentGeoType == Res) printf("%f\n", s_resCudaRC->m_time.totalKcTime);
*/

      if(j % 45 == 0 && i % 45 == 0){
        char outputpathpng[256];
        strcpy(outputpathpng, s_outputpath);
        sprintf(outputpathpng+s_outputpathsize, "\\%d.png", pngname);
        uglim_save_snapshot("cudarc", outputpathpng, "PNG", s_winx, s_winy, redraw);
        pngname++;
      }
    }
    s_camera->Rotate(angle, 1, 0, 0);
    s_camera->CenterView();
    s_canvas->Redraw();
  }

  printf("Avg.: %f\n", sumframetime / (double) numframes);
  fclose(stdout);
  printf("Finishing benchmark...\n");
  exit(0);
}

/**
* Handles keyboard input
*/
static void keyboard(int k, int st, float x, float y, void *data){

  if (st==VGL_RELEASE)
    return;

  switch (k)
  {
  case 't':
    s_displaytime = !s_displaytime;
    break;
  case 'p':
    if(s_outputpath != NULL){
      char outputpathpng[256];
      char outputpathtxt[256];

      strcpy(outputpathpng, s_outputpath);
      sprintf(outputpathpng+s_outputpathsize, ".png");

      strcpy(outputpathtxt, s_outputpath);
      sprintf(outputpathtxt+s_outputpathsize, ".txt");

      printf("Output txt file: %s\n", outputpathtxt);
      freopen(outputpathtxt,"w",stdout); 

      printf("Output png file: %s\n", outputpathpng);
      uglim_save_snapshot("cudarc", outputpathpng, "PNG", s_winx, s_winy, redraw);
    }
#ifndef CUDARC_HARC
    if(currentGeoType == Fem)
      s_femCudaRC->PrintInfo();
    if(currentGeoType == Res)
      s_resCudaRC->PrintInfo();
#endif

    if(s_outputpath)
      fclose(stdout);

    break;
  case 'h':
    s_bdryonly = !s_bdryonly;
    break;
  case 'b':
    Benchmark();
    break;

  

#ifndef CUDARC_HARC
  case 'c':
    s_probebox = !s_probebox;
    printf("Probebox: %d\n", s_probebox);
    if(currentGeoType == Fem){
      float xmin, xmax, ymin, ymax, zmin, zmax;
      s_camera->GetBox(&xmin, &xmax, &ymin, &ymax, &zmin, &zmax);

      s_femCudaRC->SetProbeBoxEnabled(s_probebox);
      s_femCudaRC->SetProbeBox(xmin + 0.3 * (xmax - xmin), xmax - 0.3 * (xmax - xmin),
                               ymin + 0.3 * (ymax - ymin), ymax - 0.3 * (ymax - ymin),
                               zmin + 0.3 * (zmax - zmin), zmax - 0.3 * (zmax - zmin));
      //s_femCudaRC->SetProbeBox(-100, 100, -100, 100, -100, 100);
    }
    if(currentGeoType == Res){
      s_resCudaRC->SetProbeBoxEnabled(s_probebox);
      s_resCudaRC->SetProbeBox(0.3, 0.7, 0.3, 0.7, 0.3, 0.7);
      //s_resCudaRC->SetProbeBox(-100, 100, -100, 100, -100, 100);
    }
    break;

  case 'd':
    s_debug = !s_debug;
    printf("Debug: %d\n", s_debug);
    if(currentGeoType == Fem)
      s_femCudaRC->SetDebugEnabled(s_debug);
    if(currentGeoType == Res)
      s_resCudaRC->SetDebugEnabled(s_debug);
    break;
  case 'm':
    s_maxedge = !s_maxedge;
    printf("Max edge length: %d\n", s_maxedge);
    if(currentGeoType == Fem)
      s_femCudaRC->SetMaxEdgeLengthEnabled(s_maxedge);
    if(currentGeoType == Res)
      s_resCudaRC->SetMaxEdgeLengthEnabled(s_maxedge);
    break;

  case 'i':
    s_isosurface = !s_isosurface;
    printf("Iso surface: %d\n", s_isosurface);
    if(currentGeoType == Fem)
      s_femCudaRC->SetIsoSurfaceEnabled(s_isosurface);
    if(currentGeoType == Res)
      s_resCudaRC->SetIsoSurfaceEnabled(s_isosurface);
    break;

  case 'v':
    s_volumetric = !s_volumetric;
    printf("Volumetric: %d\n", s_volumetric);
    if(currentGeoType == Fem)
      s_femCudaRC->SetVolumetricEnabled(s_volumetric);
    if(currentGeoType == Res)
      s_resCudaRC->SetVolumetricEnabled(s_volumetric);
    break;

  case 'r':
    s_volcolorscale = TransferFunc(s_tfpath, s_scalarmin, s_scalarmax);
    s_isocolorscale = IsoValues(s_isopath, s_scalarmin, s_scalarmax, &s_isovalues);
    
    if(currentGeoType == Fem){
      s_femCudaRC->SetVolumetricColorScale(s_volcolorscale);
      s_femCudaRC->SetIsoColorScale(s_isocolorscale, s_isovalues);
    }
    if(currentGeoType == Res){
      s_resCudaRC->SetVolumetricColorScale(s_volcolorscale);
      s_resCudaRC->SetIsoColorScale(s_isocolorscale, s_isovalues);
    }
    break;
  
  case '0':
    printf("Interpol type: const\n");
    currentInterpolType = Const;
    break;

  case '1':
    printf("Interpol type: linear/trilinear, fetching from texture\n");
    currentInterpolType = Linear;
    break;

  case '2':
    printf("Interpol type: linear/trilinear, using quadrature\n");
    currentInterpolType = Quad;
    break;

  case '3':
    printf("Interpol type: fixed steps\n");
    currentInterpolType = Step;
    break;

  case 'a':
    if(currentNumTraverses > 0){
      currentNumTraverses--;
      printf("Num traverses: %d\n", currentNumTraverses);
    }
    break;

  case 's':
    currentNumTraverses++;
    printf("Num traverses: %d\n", currentNumTraverses);
    break;

  case 'w':
    currentNumPeeling++;
    printf("Current peeling: %d\n", currentNumPeeling);
    break;

  case 'q':
    if(currentNumPeeling > 0){
      currentNumPeeling--;
      printf("Current peeling: %d\n", currentNumPeeling);
    }
    break;

  case 'y':
   // if(currentDelta > 0){
      currentDelta-=0.0005;
      printf("Current delta: %f\n", currentDelta);
    //}
    break;

  case 'u':
    currentDelta+=0.0005;
    printf("Current delta: %f\n", currentDelta);
    break;

  case 'f':
   // if(currentDelta > 0){
      currentDeltaW-=5;
      printf("Current deltaW: %f\n", currentDeltaW);
    //}
    break;

  case 'g':
    currentDeltaW+=5;
    printf("Current deltaW: %f\n", currentDeltaW);
    break;


  case 'l':
    currentZero+=0.002f;
    printf("Current zero: %f\n", currentZero);
    break;

  case 'k':
   // if(currentDelta > 0){
      currentZero-=0.002f;
      printf("Current zero: %f\n", currentZero);
    //}
    break;


  case '-':
    if(currentNumSteps > 1){
      currentNumSteps--;
      printf("Num steps: %d\n", currentNumSteps);
    }
    break;

  case '=':
    currentNumSteps++;
    printf("Num steps: %d\n", currentNumSteps);
    break;

  case '[':
    if(currentNumSteps > 10){
      currentNumSteps-=10;
      printf("Num steps: %d\n", currentNumSteps);
    }
    break;

  case ']':
    currentNumSteps+=10;
    printf("Num steps: %d\n", currentNumSteps);
    break;

  case ',':
    if(currentNumSteps > 100){
      currentNumSteps-=100;
      printf("Num steps: %d\n", currentNumSteps);
    }
    break;

  case '.':
    currentNumSteps+=100;
    printf("Num steps: %d\n", currentNumSteps);
    break;
  }
  if(currentGeoType == Fem){
    s_femCudaRC->SetInterpolationType(currentInterpolType);
    s_femCudaRC->SetNumSteps(currentNumSteps);
    s_femCudaRC->SetNumTraverses(currentNumTraverses);
    s_femCudaRC->SetNumPeeling(currentNumPeeling);
  }
  if(currentGeoType == Res){
    s_resCudaRC->SetInterpolationType(currentInterpolType);
    s_resCudaRC->SetNumSteps(currentNumSteps);
    s_resCudaRC->SetNumTraverses(currentNumTraverses);
    s_resCudaRC->SetNumPeeling(currentNumPeeling);
  }
  
#else
  }
#endif




}

/**
* Create a TpvProperty from a TopModel
*/
static TpvProperty* CreateTpvPropertyFromTopModel(TopModel* topmodel, double* scalars, int baseId)
{

  int numNodes = topmodel->GetNNodes();
  float* floatScalars = new float[numNodes];

  s_scalarmin = FLT_MAX; s_scalarmax = -FLT_MAX;  
  for (int i = 0; i < numNodes; i++) {
    float s = scalars[i];
    s_scalarmin = s < s_scalarmin ? s : s_scalarmin;
    s_scalarmax = s > s_scalarmax ? s : s_scalarmax;
    floatScalars[i] = s;
  }
  bool takeownership;
  TpvProperty* prop = new TpvProperty(TpvProperty::PROP_NODE, 1);
  prop->SetBaseGlobalId(baseId);
  prop->SetValues(numNodes, floatScalars, takeownership=true);
  prop->SetMinValue(&s_scalarmin);
  prop->SetMaxValue(&s_scalarmax);
  prop->SetIndirection(NULL);

  s_volcolorscale = TransferFunc(s_tfpath, s_scalarmin, s_scalarmax);
  s_isocolorscale = IsoValues(s_isopath, s_scalarmin, s_scalarmax, &s_isovalues);

  return prop;
}

/**
* Load prop from a file
*/
static double* LoadTpvPropertyFile(TopModel* topmodel, const char* propfile){

  int numnodes = topmodel->GetNNodes();
  FILE* fpscalar = fopen(propfile, "r");
  double* scalars = new double[numnodes];

  if(fpscalar == NULL){
    printf("Prop file not found\n");
    exit(1);
  }

  printf("Loading prop file: %s \n", propfile);
  for (int i = 0; i < numnodes; i++){
    float aux;
    fscanf(fpscalar, " %f", &aux);
    scalars[i] = (double) aux;
  }

  return scalars;
}

/**
* Loads a Plot3d model
*/
void InitPlot3d(const char* fullPath, const char* propfile){

#ifdef CUDARC_VERBOSE
  printf("Loading Plot3d model %s ... ", fullPath);
#endif

  std::string strFullPath = std::string(fullPath);

  int pos = strFullPath.rfind(".");
  std::string strWoExtension = strFullPath.substr(0, pos);
  std::string strGrid = std::string(strWoExtension); strGrid.append(".grid", 5);
  std::string strSol = std::string(strWoExtension); strSol.append(".solution", 9);

  int posslash = strFullPath.rfind("\\");
  int posinvslash = strFullPath.rfind("/");
  if(posslash < posinvslash)
    posslash = posinvslash;
  std::string strDir = strFullPath.substr(0, posslash+1);

  s_topmodel = new TopModel();
#ifdef CUDARC_HEX
  TopPlot3DReader::Read(s_topmodel, strGrid.c_str(), strSol.c_str(), true);
#else
  TopPlot3DReader::Read(s_topmodel, strGrid.c_str(), strSol.c_str(), false);
#endif

  double xmin, xmax, ymin, ymax, zmin, zmax; 
  TopUtil::ComputeBoundingBox(s_topmodel, &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);
  s_bbMin.x = xmin; s_bbMin.y = ymin; s_bbMin.z = zmin;
  s_bbMax.x = xmax; s_bbMax.y = ymax; s_bbMax.z = zmax;

  //Scalars
  double* scalars = NULL;
  if(propfile == NULL){
    scalars = new double[s_topmodel->GetNNodes()];
    for(int i=1; i <= s_topmodel->GetNNodes(); i++){
      TopNode node = s_topmodel->GetNodeAtId(i);
      double* aux = (double *)s_topmodel->GetAttrib(node);
      scalars[i-1] = *aux;
    }
  }
  else
    scalars = LoadTpvPropertyFile(s_topmodel, strDir.append(propfile).c_str());

  //Tpv
  s_tpvFemTetraGeometry = new TpvFemTetraGeometry3();
  s_tpvFemTetraGeometry->SetModel(s_topmodel);
  s_tpvFemTetraGeometry->SetExtractTetrahedronVertexIncidences(CUDARC_EXTRACT_TET_VERT_INCIDENCES);

  s_tpvproperty = CreateTpvPropertyFromTopModel(s_topmodel, scalars, 1);

#ifdef CUDARC_VERBOSE
  printf("Done.\n");
#endif
}

/**
* Loads a top model
*/
void InitTopModel(const char* fullPath, const char* propfile){
#ifdef CUDARC_VERBOSE
  printf("Loading Top model %s ... ", fullPath);
#endif

  std::string strFullPath = std::string(fullPath);

  int posdot = strFullPath.rfind(".");
  std::string strWoExtension = strFullPath.substr(0, posdot);

  int posslash = strFullPath.rfind("\\");
  int posinvslash = strFullPath.rfind("/");
  if(posslash < posinvslash)
    posslash = posinvslash;
  std::string strDir = strFullPath.substr(0, posslash+1);

  s_topmodel = new TopModel();
  TopModelReader* topModelReader = new TopModelReader(s_topmodel, strWoExtension.c_str());
  topModelReader->SetElemAndNodeIdsEnabled(true);
  topModelReader->Open();
  topModelReader->Read();
  //TODO: check memory leak
  //topModelReader->Close();
  //delete topModelReader;

  //Set topmodel elements ids
  int id=1;
  for(TopModel::ElemItr itr(s_topmodel); itr.IsValid(); itr.Next()){
    TopElement el = itr.GetCurr();
    s_topmodel->SetId(el, id);
    id++;
  }

  double xmin, xmax, ymin, ymax, zmin, zmax; 
  TopUtil::ComputeBoundingBox(s_topmodel, &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);
  s_bbMin.x = xmin; s_bbMin.y = ymin; s_bbMin.z = zmin;
  s_bbMax.x = xmax; s_bbMax.y = ymax; s_bbMax.z = zmax;


  //Tpv
  s_tpvFemTetraGeometry = new TpvFemTetraGeometry3();
  s_tpvFemTetraGeometry->SetModel(s_topmodel);
  s_tpvFemTetraGeometry->SetExtractTetrahedronVertexIncidences(CUDARC_EXTRACT_TET_VERT_INCIDENCES);

  double* scalars = NULL;
  if(propfile == NULL){
    scalars = new double[s_topmodel->GetNNodes()];
    topModelReader->GetResultData(0, 0, 0, 0, scalars);
  }
  else
    scalars = LoadTpvPropertyFile(s_topmodel, strDir.append(propfile).c_str());

  s_tpvproperty = CreateTpvPropertyFromTopModel(s_topmodel, scalars, 0);


#ifdef CUDARC_VERBOSE
  printf("Done.\n");
#endif
}

/**
* Create a tpvproperty from a resproperty
*/
static TpvProperty* CreateTpvPropertyFromResProperty (TopMultiModel* multimodel, ResProperty* resprop)
{
  float* node_prop = resprop->CreateSmoothProperty(resprop->GetCurrentStep());
  s_scalarmin = FLT_MAX; s_scalarmax = -FLT_MAX;
  int numnodes = multimodel->GetTotalNNodes();
  int aux = 0;
  for (int i = 0; i < numnodes; i++) {
    if (node_prop[i] != resprop->GetNull()) {
      float s = node_prop[i];
      s_scalarmin = s < s_scalarmin ? s : s_scalarmin;
      s_scalarmax = s > s_scalarmax ? s : s_scalarmax;
      aux++;
    }
  }
  bool takeownership;
  TpvProperty* prop = new TpvProperty(TpvProperty::PROP_NODE, 1);
  prop->SetBaseGlobalId(0);
  prop->SetValues(numnodes, node_prop, takeownership=true);
  prop->SetMinValue(&s_scalarmin);
  prop->SetMaxValue(&s_scalarmax);
  prop->SetIndirection(NULL);

  s_volcolorscale = TransferFunc(s_tfpath, s_scalarmin, s_scalarmax);
  s_isocolorscale = IsoValues(s_isopath,s_scalarmin, s_scalarmax, &s_isovalues);

  return prop;
}

/**
* Loads resmodel
*/
void InitRes(const char* fullPath){

#ifdef CUDARC_VERBOSE
  printf("Loading PPOPSP model %s ... ", fullPath);
#endif

  gposmesh_init(NULL);
  gpospposp_init();
  gposcmg_init("../../../app/geresim/data/dcta200810.dat");
  gposecl_init();

  ResOpenOptions options;
  bool ignore = true;
  ResSetBooleanOption(&options, "IgnoreGeometryFile", &ignore);
  ResModelNew* resmodelnew = ResModelOpen(fullPath, 1, &fullPath, NULL);

  ResModel* resmodel = resmodelnew->GetModel();
  s_resGeometry = resmodel->GetGeometry();
  s_resproperty = resmodel->GetProperty("SO");
  s_resproperty->SetStep(resmodel->GetStep(0));  
  //TODO: Discrete?
  s_resproperty->SetDiscrete();

  s_bbMax.x = s_resGeometry->ActXmax(); s_bbMax.y = s_resGeometry->ActYmax(); s_bbMax.z = s_resGeometry->ActZmax(); 
  s_bbMin.x = s_resGeometry->ActXmin(); s_bbMin.y = s_resGeometry->ActYmin(); s_bbMin.z = s_resGeometry->ActZmin(); 

  //Tpv
  s_tpvResTetraGeometry = new TpvResTetraGeometry3();
  s_tpvResTetraGeometry->SetModel(s_resGeometry->GetModel());
  s_tpvResTetraGeometry->SetExtractTetrahedronVertexIncidences(CUDARC_EXTRACT_TET_VERT_INCIDENCES);
  s_tpvResTetraGeometry->SetElemVis(s_resGeometry->GetVisibleCells());

  s_topmultimodel = s_resGeometry->GetModel();
  s_tpvproperty = CreateTpvPropertyFromResProperty(s_topmultimodel, s_resproperty);

#ifdef CUDARC_VERBOSE
  printf("Done.\n");
#endif
}

/**
* Initialize OpenGL, camera and canvas
*/
void InitGL(int argc, char **argv){

  glutInit(&argc, argv);
  glutInitWindowSize(s_winx, s_winy);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
  int window = glutCreateWindow("CUDA RC");

  ugl_init();

  s_camera = new VglCamera();
  s_camera->SetOrtho(false);
  s_camera->SetHeadlight(GL_LIGHT0);
  s_camera->SetAngle(60);
  s_camera->SetAutoFit(true);
  s_camera->SetBox(s_bbMin.x, s_bbMax.x,
    s_bbMin.y, s_bbMax.y,
    s_bbMin.z, s_bbMax.z);

  s_canvas = new VglGlutCanvas(window, s_camera);
  s_canvas->SetRedrawFunc(redraw, NULL);
  s_canvas->AddIdle(redraw, NULL, VGL_FOREVER);

  VglModeHandler* modehandler = new VglModeHandler(s_canvas);
  modehandler->AddHandler(new VglManipHandler(s_canvas), 'm');
  modehandler->AddHandler(new VglObjectHandler(s_canvas), 'o');
  modehandler->AddHandler(new VglNavigHandler(s_canvas), 'n');
  modehandler->AddHandler(new VglZoomHandler(s_canvas), 'z');
  modehandler->ChangeHandler('m');

  s_canvas->AddHandler(modehandler);

  s_canvas->SetKeyboardFunc(keyboard,NULL);

  s_canvas->Activate();

}

/**
* Main function, entry point
*/
int main(int argc, char **argv){

#ifdef CUDARC_TIME
  s_displaytime = true;
#endif

  if (argc < 2){
    fprintf(stderr, "usage: %s modelpath [-shaderpath] [-blocksize] [-maxpeel] [-tfsize] [-winx] [-winy] [-scalez] [-transp] [-outputpath] \n", argv[0]);
    exit(1);
  }


  s_modelpath = argv[1];
  for(int i=2; i<argc; i++){
    if(strcmp(argv[i], "-shaderpath") == 0)
      s_shaderpath = argv[i+1];
    if(strcmp(argv[i], "-propfile") == 0)
      s_propfile = argv[i+1];
    else if(strcmp(argv[i], "-blocksize") == 0)
      s_blocksize = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-maxpeel") == 0)
      s_maxpeel = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-tfsize") == 0)
      s_tfsize = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-tfpath") == 0)
      s_tfpath = argv[i+1];
    else if(strcmp(argv[i], "-winx") == 0)
      s_winx = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-winy") == 0)
      s_winy = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-scalez") == 0)
      s_scalez = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-transp") == 0)
      s_transp = atof(argv[i+1]);
    else if(strcmp(argv[i], "-outputpath") == 0)
      s_outputpath = argv[i+1];
    else if(strcmp(argv[i], "-zetapsigammasize") == 0)
      s_zetapsigammasize = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-zetapsigammapath") == 0)
      s_zetapsigammapath = argv[i+1];
    else if(strcmp(argv[i], "-bdryonly") == 0)
      s_bdryonly = atof(argv[i+1]);
    else if(strcmp(argv[i], "-maxedge") == 0)
      s_maxedge = atof(argv[i+1]);
    else if(strcmp(argv[i], "-tessellation") == 0)
      s_tessellation = atof(argv[i+1]);
    else if(strcmp(argv[i], "-normalizefield") == 0)
      s_normalizefield = atof(argv[i+1]);
    else if(strcmp(argv[i], "-interpol") == 0){
      char* interpoltype = argv[i+1];
      if(strcmp(interpoltype, "const") == 0)
        currentInterpolType = Const;
      else if(strcmp(interpoltype, "linear") == 0 || strcmp(interpoltype, "trilinear") == 0)
        currentInterpolType = Linear;
      else if(strcmp(interpoltype, "quad") == 0)
        currentInterpolType = Quad;
      else if(strcmp(interpoltype, "step") == 0)
        currentInterpolType = Step;
    }
    else if(strcmp(argv[i], "-avg") == 0){
      s_averageframes = atoi(argv[i+1]);
      s_displaytime = false;
    }
    else if(strcmp(argv[i], "-benchmark") == 0){
      s_benchmark = true; 
      s_displaytime = false;
      s_averageframes = 1;
    }
  }

  const char* fileExtension = strrchr(s_modelpath, '.'); 

#ifndef CUDARC_HARC
  if(strcmp(fileExtension, ".PPOSP") == 0 || strcmp(fileExtension, ".pposp") == 0){
    InitRes(s_modelpath);
    currentGeoType = Res;
    s_resCudaRC = new CudaRC<TpvResGeometryModel>();
    s_resCudaRC->SetGeometry(s_tpvResTetraGeometry);
    s_resCudaRC->SetResGeometry(s_resGeometry);
    s_resCudaRC->SetModel(s_topmultimodel);
    s_resCudaRC->SetVolumetricColorScale(s_volcolorscale);
    s_resCudaRC->SetIsoColorScale(s_isocolorscale, s_isovalues);
    s_resCudaRC->SetProperty(s_tpvproperty);
    s_resCudaRC->SetDebugEnabled(false);
    s_resCudaRC->SetExplodedView(false);
    s_resCudaRC->SetMaxEdgeLengthEnabled(s_maxedge);
    s_resCudaRC->SetInterpolationType(currentInterpolType);
    s_resCudaRC->SetNumSteps(currentNumSteps);
    s_resCudaRC->SetNumTraverses(currentNumTraverses);
    s_resCudaRC->SetNumPeeling(currentNumPeeling);
    s_resCudaRC->SetBlockSize(s_blocksize, s_blocksize);
    s_resCudaRC->SetWindowSize(s_winx, s_winy);
    s_resCudaRC->SetMaxNumPeel(s_maxpeel);
    s_resCudaRC->SetTessellation(s_tessellation);
    s_resCudaRC->SetNormalizedField(s_normalizefield);
    s_resCudaRC->SetPaths(s_zetapsigammasize, s_zetapsigammapath, s_shaderpath);
    InitGL(argc, argv);
  }
  if(strcmp(fileExtension, ".ele") == 0 || strcmp(fileExtension, ".node") == 0){
    InitTopModel(s_modelpath, s_propfile);
    currentGeoType = Fem;
    s_femCudaRC = new CudaRC<TpvFemGeometryModel>();
    s_femCudaRC->SetGeometry(s_tpvFemTetraGeometry);
    s_femCudaRC->SetModel(s_topmodel);
    s_femCudaRC->SetVolumetricColorScale(s_volcolorscale);
    s_femCudaRC->SetIsoColorScale(s_isocolorscale, s_isovalues);
    s_femCudaRC->SetProperty(s_tpvproperty);
    s_femCudaRC->SetDebugEnabled(false);
    s_femCudaRC->SetExplodedView(false);
    s_femCudaRC->SetMaxEdgeLengthEnabled(s_maxedge);
    s_femCudaRC->SetInterpolationType(currentInterpolType);
    s_femCudaRC->SetNumSteps(currentNumSteps);
    s_femCudaRC->SetNumTraverses(currentNumTraverses);
    s_femCudaRC->SetNumPeeling(currentNumPeeling);
    s_femCudaRC->SetBlockSize(s_blocksize, s_blocksize);
    s_femCudaRC->SetWindowSize(s_winx, s_winy);
    s_femCudaRC->SetMaxNumPeel(s_maxpeel);
    s_femCudaRC->SetTessellation(s_tessellation);
    s_femCudaRC->SetNormalizedField(s_normalizefield);
    s_femCudaRC->SetPaths(s_zetapsigammasize, s_zetapsigammapath, s_shaderpath);
    InitGL(argc, argv);
  }

  if(strcmp(fileExtension, ".grid") == 0 || strcmp(fileExtension, ".solution") == 0){
    InitPlot3d(s_modelpath, s_propfile);
    currentGeoType = Fem;
    s_femCudaRC = new CudaRC<TpvFemGeometryModel>();
    s_femCudaRC->SetGeometry(s_tpvFemTetraGeometry);
    s_femCudaRC->SetModel(s_topmodel);
    s_femCudaRC->SetVolumetricColorScale(s_volcolorscale);
    s_femCudaRC->SetIsoColorScale(s_isocolorscale, s_isovalues);
    s_femCudaRC->SetProperty(s_tpvproperty);
    s_femCudaRC->SetDebugEnabled(false);
    s_femCudaRC->SetExplodedView(false);
    s_femCudaRC->SetMaxEdgeLengthEnabled(s_maxedge);
    s_femCudaRC->SetInterpolationType(currentInterpolType);
    s_femCudaRC->SetNumSteps(currentNumSteps);
    s_femCudaRC->SetNumTraverses(currentNumTraverses);
    s_femCudaRC->SetNumPeeling(currentNumPeeling);
    s_femCudaRC->SetBlockSize(s_blocksize, s_blocksize);
    s_femCudaRC->SetWindowSize(s_winx, s_winy);
    s_femCudaRC->SetMaxNumPeel(s_maxpeel);
    s_femCudaRC->SetTessellation(s_tessellation);
    s_femCudaRC->SetNormalizedField(s_normalizefield);
    s_femCudaRC->SetPaths(s_zetapsigammasize, s_zetapsigammapath, s_shaderpath);
    InitGL(argc, argv);
  }
#else
  if(strcmp(fileExtension, ".PPOSP") == 0 || strcmp(fileExtension, ".pposp") == 0){
    s_resHarc = new TpvHARCVolRenderer3<TpvResGeometryModel>();
    InitRes(s_modelpath);
    currentGeoType = Res;
    s_resHarc->SetGeometry(s_tpvResTetraGeometry);
    s_resHarc->SetColorScale(s_volcolorscale);
    s_resHarc->SetProperty(s_tpvproperty);
    s_resHarc->SetShaderPath("../../../src/topsview/renderer/cg");
    s_resHarc->SetIsosurfacesEnabled(false);
    s_resHarc->SetMaxEdgeLengthEnabled(s_maxedge);
    s_resHarc->SetNormalizedField(s_normalizefield);
    InitGL(argc, argv);
  }
  else if(strcmp(fileExtension, ".grid") == 0 || strcmp(fileExtension, ".solution") == 0){
    s_femHarc = new TpvHARCVolRenderer3<TpvFemGeometryModel>();
    InitPlot3d(s_modelpath, s_propfile);
    currentGeoType = Fem;
    s_femHarc->SetGeometry(s_tpvFemTetraGeometry);
    s_femHarc->SetColorScale(s_volcolorscale);
    s_femHarc->SetProperty(s_tpvproperty);
    s_femHarc->SetShaderPath("../../../src/topsview/renderer/cg");
    s_femHarc->SetIsosurfacesEnabled(false);
    s_femHarc->SetMaxEdgeLengthEnabled(s_maxedge);
    s_femHarc->SetNormalizedField(s_normalizefield);
    InitGL(argc, argv);
  }
  else if(strcmp(fileExtension, ".node") == 0 || strcmp(fileExtension, ".ele") == 0){
    s_femHarc = new TpvHARCVolRenderer3<TpvFemGeometryModel>();
    InitTopModel(s_modelpath, s_propfile);
    currentGeoType = Fem;
    s_femHarc->SetGeometry(s_tpvFemTetraGeometry);
    s_femHarc->SetColorScale(s_volcolorscale);
    s_femHarc->SetProperty(s_tpvproperty);
    s_femHarc->SetShaderPath("../../../src/topsview/renderer/cg");
    s_femHarc->SetIsosurfacesEnabled(false);
    s_femHarc->SetMaxEdgeLengthEnabled(s_maxedge);
    s_femHarc->SetNormalizedField(s_normalizefield);
    InitGL(argc, argv);
  }
#endif

  //Outputpath
  if(s_outputpath != NULL){
    _mkdir(s_outputpath);

    s_outputpathsize = strlen(s_outputpath);

/*
#ifdef _DEBUG
    s_outputpathsize += sprintf(s_outputpath+s_outputpathsize, "\\debug_");
#else
    s_outputpathsize += sprintf(s_outputpath+s_outputpathsize, "\\");
#endif
#ifdef CUDARC_HARC
    s_outputpathsize += sprintf(s_outputpath+s_outputpathsize, "harc");
#else
#ifdef CUDARC_HEX
    s_outputpathsize += sprintf(s_outputpath+s_outputpathsize, "hex");
#else
    s_outputpathsize += sprintf(s_outputpath+s_outputpathsize, "tet");
#endif
#ifdef CUDARC_CONST
    s_outputpathsize += sprintf(s_outputpath+s_outputpathsize, "_const");
#else
#ifdef CUDARC_INTEGRATE_FIXEDSTEPS
    char buffer[4];
    itoa(CUDARC_NUM_STEPS, buffer, 10);
    s_outputpathsize += sprintf(s_outputpath+s_outputpathsize, "_fixed");
    s_outputpathsize += sprintf(s_outputpath+s_outputpathsize, buffer);
#else
#ifdef CUDARC_CALCULATE_ZETAPSI
    s_outputpathsize += sprintf(s_outputpath+s_outputpathsize, "_gauss");
#else
    s_outputpathsize += sprintf(s_outputpath+s_outputpathsize, "_pre");
#endif
#endif
#endif
#endif
    */
  }
 
  glutMainLoop();

}