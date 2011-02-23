
#define CUDARC_MAX_MESH_TEXTURE_WIDTH 1
#define CUDARC_MAX_ITERATIONS 2000

//#define CUDARC_WINGL
//#define CUDARC_HARC
#define CUDARC_VERBOSE
#define CUDARC_TIME
#define CUDARC_EXTRACT_TET_VERT_INCIDENCES false
//#define CUDARC_PROBE_BOX
#define CUDARC_ISOSURFACE
#define CUDARC_CUTTING_PLANE
//#define CUDARC_GRADIENT_PERVERTEX //(tet. only!)
//#define CUDARC_WHITE
//#define CUDARC_HEX
//#define CUDARC_PLUCKER //(tet. only!)
//#define CUDARC_BILINEAR //(hex. only!)


//Undefs according to the tet./hex.
#ifdef CUDARC_HEX
#undef CUDARC_PLUCKER
#undef CUDARC_GRADIENT_PERVERTEX
#else
#undef CUDARC_BILINEAR
#endif