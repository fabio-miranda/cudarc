/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* cudarc.cu: CUDA functions
*/

#define EPSILON 1e-6

#ifndef CUDARC_WINGL
	#include <windows.h>
#else
	#include <GL/glut.h>
#endif

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include <math_constants.h>

#include "defines.h"
#include "cudarc.cuh"
#include "cudamemory.cuh"

enum InterpolType {Const = 0, Linear = 1, Quad = 2, Step = 3};

struct Elem{
  float4 interpolfunc0;
  float4 adj0;
#ifdef CUDARC_HEX
  float4 interpolfunc1;
  float4 adj1;
#endif

};

struct Ray{
  float4 acccolor;
  float4 dir;
  float4 eyepos;
  float t;
  float frontid;
  float frontface;
  float frontscalar;
  Elem currentelem;
};
/*
inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
*/
inline __host__ __device__ float4 cross(float4 a, float4 b)
{ 
  return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0); 
}


inline __host__ __device__ float permuted_inner_produtct(float4 pr, float4 qr, float4 ps, float4 qs)
{
  return dot(pr, qs) + dot(qr, ps);
}

inline __device__ float intersect_ray_plane(float4 eyepos, float4 dir, float4 normal)
{
  //t = -(P0 . N + d) / (V . N) (http://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm)
    //t = -(eyePos . normal + d) / (eyeDir . normal)
  float samedir = dot(dir, normal);
  if(samedir > 0)
    return - dot(normal, eyepos) / samedir;
  return CUDART_INF_F;
}

inline __device__ float intersect_ray_probeplane(float4 eyepos, float4 dir, float4 normal, float* tnear, float* tfar)
{
  float samedir = dot(dir, normal);
  if(samedir > 0)
    *tfar = fminf(- dot(normal, eyepos) / samedir, *tfar);
  else if(samedir < 0)
    *tnear = fmaxf(- dot(normal, eyepos) / samedir, *tnear);
}


#ifdef CUDARC_HEX

/**
* Ray bilinear patch intersection (hexahedral mesh)
*/

__device__ float ComputeU(float v, float a1, float a2, float b1, float b2, float c1, float c2, float d1, float d2){
  float a = v * a2 + b2;
  float b = v * (a2 - a1) + b2 - b1;

  if(fabs(b) > fabs(a))
    return (v * (c1 - c2) + d1 - d2) / b;
  else
    return (- v * c2 - d2) / a;
}

__device__ float ComputeT(Ray* ray, float4 p){
  /*
  if(fabs(ray->dir.x) >= fabs(ray->dir.y) && fabs(ray->dir.x) >= fabs(ray->dir.z))
  return (p.x - ray->eyepos.x) / ray->dir.x;
  else if(fabs(ray->dir.y) >= fabs(ray->dir.z))
  return (p.y - ray->eyepos.y) / ray->dir.y;
  else
  return (p.z - ray->eyepos.z) / ray->dir.z;
  */
  //p.w = 1;
  //ray->eyepos.w = 1;
  return length(p - ray->eyepos);
} 

__device__ float Solve(Ray* ray, float4 v00, float4 v01, float4 v10, float4 v11, float v, float a1, float a2, float b1, float b2, float c1, float c2, float d1, float d2){

  if(v >= -EPSILON && v <= 1.0f + EPSILON){
    float u = ComputeU(v, a1, a2, b1, b2, c1, c2, d1, d2);
    if(u >= -EPSILON && u <= 1.0f + EPSILON){
      float4 p = (1.0f - u) * (1.0f - v) * v00 + v * (1.0f - u) * v01 + u * (1.0f - v) * v10 + u * v * v11;
      float t = ComputeT(ray, p);
      //if(t >= -EPSILON)
        return t;
      //return length(p - ray->eyepos);
    }
  }
  //return u;
  return CUDART_INF_F;

}

__device__ float2 IntersectBilinear(Ray* ray, float4 v00, float4 v01, float4 v10, float4 v11, bool getfart){



  float4 a = v11 - v10 - v01 + v00;
  float4 b = v10 - v00;
  float4 c = v01 - v00;
  float4 d = v00 - ray->eyepos;

  float a1 = a.x * ray->dir.z - a.z * ray->dir.x;
  float b1 = b.x * ray->dir.z - b.z * ray->dir.x;
  float c1 = c.x * ray->dir.z - c.z * ray->dir.x;
  float d1 = d.x * ray->dir.z - d.z * ray->dir.x;
  float a2 = a.y * ray->dir.z - a.z * ray->dir.y;
  float b2 = b.y * ray->dir.z - b.z * ray->dir.y;
  float c2 = c.y * ray->dir.z - c.z * ray->dir.y;
  float d2 = d.y * ray->dir.z - d.z * ray->dir.y;

  //Solve the equation (A2C1 - A1C2) * v^2 + (A2D1 - A1D2 + B2C1 - B1C2) * v + (B2D1 - B1D2) = 0
  float aux_a = a2 * c1 - a1 * c2;
  float aux_b = a2 * d1 - a1 * d2 + b2 * c1 - b1 * c2;
  float aux_c = b2 * d1 - b1 * d2;
  float delta = aux_b * aux_b - 4.0f * (aux_a) * (aux_c);

#if 0
  //a close to zero
  //if(aux_a == 0.0f){
  if(aux_a >= -EPSILON && aux_a <= EPSILON){
    //if(aux_b != 0.0f){
    if(aux_b <= -EPSILON || aux_b >= EPSILON){
      float root = -aux_c / aux_b;
      if(root > -EPSILON && root < 1 + EPSILON){
        //return 1;
        //return Solve(ray, v00, v01, v10, v11, root, a1, a2, b1, b2, c1, c2, d1, d2);
        return make_float2(Solve(ray, v00, v01, v10, v11, root, a1, a2, b1, b2, c1, c2, d1, d2), 1); //cai aqui
      }
      else{
        //return 0;
        //return CUDART_INF_F;
        return make_float2(CUDART_INF_F, 0);
      }
    }
    else{
      //return 0;
      //return CUDART_INF_F;
      return make_float2(CUDART_INF_F, 0);
    }
  }

  if(delta <= EPSILON){
    if(delta >= -EPSILON && delta <= EPSILON){
      float root = -aux_b / aux_a;
      if(root > -EPSILON && root < 1 + EPSILON){
        //return 1;
        //return Solve(ray, v00, v01, v10, v11, root, a1, a2, b1, b2, c1, c2, d1, d2);
        return make_float2(Solve(ray, v00, v01, v10, v11, root, a1, a2, b1, b2, c1, c2, d1, d2), 1);
      }
      else{
        //return 0;
        //return CUDART_INF_F;
        return make_float2(CUDART_INF_F, 0);
      }
    }
    else{
      //return 0;
      //return CUDART_INF_F;
      return make_float2(CUDART_INF_F, 0);
    }
  }

  float q;
  if(aux_b < EPSILON)
    q = - 0.5f * (aux_b - sqrtf(delta));
  else
    q = - 0.5f * (aux_b + sqrtf(delta));

  float root1 = q / aux_a;
  float root2 = aux_c / q;

  if(root1 > -EPSILON && root1 < 1 + EPSILON && root2 > -EPSILON && root2 < 1 + EPSILON){
    //return 2;
    float t1 = Solve(ray, v00, v01, v10, v11, root1, a1, a2, b1, b2, c1, c2, d1, d2);
    float t2 = Solve(ray, v00, v01, v10, v11, root2, a1, a2, b1, b2, c1, c2, d1, d2);

    //return fminf(t1, t2);
    return make_float2(fminf(t1, t2), 2);
  }
  else if(root1 > -EPSILON && root1 < 1 + EPSILON){
    //return 1;

    //return Solve(ray, v00, v01, v10, v11, root1, a1, a2, b1, b2, c1, c2, d1, d2);
    return make_float2(Solve(ray, v00, v01, v10, v11, root1, a1, a2, b1, b2, c1, c2, d1, d2), 2);
  }
  else if(root2 > -EPSILON && root2 < 1 + EPSILON){
    //return 1;

    //return Solve(ray, v00, v01, v10, v11, root2, a1, a2, b1, b2, c1, c2, d1, d2);
    return make_float2(Solve(ray, v00, v01, v10, v11, root2, a1, a2, b1, b2, c1, c2, d1, d2), 2);
  }
  else{
    //return 0;
    //return CUDART_INF_F;
    return make_float2(CUDART_INF_F, 0);
  }

#else
  
  if(delta < -EPSILON){
    return make_float2(CUDART_INF_F, 0);
    //return 0;
    //return CUDART_INF_F;
  }
  else if(delta >= -EPSILON && delta <= EPSILON){
    //else if(delta == 0){
    float t = Solve(ray, v00, v01, v10, v11, -aux_b / (2.0f * aux_a), a1, a2, b1, b2, c1, c2, d1, d2);

    if(t + EPSILON < ray->t)
      return make_float2(CUDART_INF_F, 1);
    else
      return make_float2(t, 1);
    
    //return make_float2(t, 1);
    //return 1;
    //return t;
  }
  else{
    
    //float v1 = (- aux_b + sqrtf(delta)) / (2.0f * aux_a);
    //float v2 = (- aux_b - sqrtf(delta)) / (2.0f * aux_a);
    float q;
    if(aux_b < 0.0f)
      q = - 0.5f * (aux_b - sqrtf(delta));
    else
      q = - 0.5f * (aux_b + sqrtf(delta));

    float v1 = q / aux_a;
    float v2 = aux_c / q;

    float t1 = Solve(ray, v00, v01, v10, v11, v1, a1, a2, b1, b2, c1, c2, d1, d2);
    float t2 = Solve(ray, v00, v01, v10, v11, v2, a1, a2, b1, b2, c1, c2, d1, d2);
    
    if(t1 + EPSILON < ray->t)
      t1 = CUDART_INF_F;
    if(t2 + EPSILON < ray->t)
      t2 = CUDART_INF_F;
    

    if(getfart)
      return make_float2(fmaxf(t1, t2), 2);
    else
      return make_float2(fminf(t1, t2), 2);

    /*
    if(t2 == CUDART_INF_F)
    return make_float2(t1, 1);
    else if(t1 == CUDART_INF_F)
    return make_float2(t2, 1);
    else
    return make_float2(tmin, 2);
    */
    //return 2;


    /*
    if(tmin < CUDART_INF_F)
    return tmin;
    else
    return -1;

    if(t1 < CUDART_INF_F && t2 < CUDART_INF_F)
    return 2;
    else if(t1 < CUDART_INF_F)
    return 1;
    else if(t2 < CUDART_INF_F)
    return 1;
    else
    return 0;
    */
  }
#endif
  /*
  if(delta == 0){
  float v = -aux_b / (2 * aux_a);
  return Solve(ray, v00, v01, v10, v11, v, a1, a2, b1, b2, c1, c2, d1, d2);
  }
  return 0;
  */

}

/**
* Find scalar of the (x,y,z) point (hexahedral mesh)
*/
inline __device__ float FindScalar(Ray* ray, float p_t){

  float4 pos = ray->eyepos + p_t * ray->dir;
  pos.w = 1.0;

  float4 interpolfunc0 = ray->currentelem.interpolfunc0;
  float4 interpolfunc1 = ray->currentelem.interpolfunc1;

  return interpolfunc0.x * pos.x + interpolfunc0.y * pos.y + interpolfunc0.z * pos.z + interpolfunc0.w * pos.x * pos.y
    + interpolfunc1.x * pos.x * pos.z + interpolfunc1.y * pos.y * pos.z + interpolfunc1.z * pos.x * pos.y * pos.z + interpolfunc1.w;


}


/**
* Calculate ZetaPsi, using gaussian quadrature (hexahedral mesh)
*/

#define REAL float
inline __device__ REAL Integral(REAL t, REAL tb, REAL tf, REAL D, REAL pb, REAL pf, REAL sb, REAL sf, REAL w0, REAL w1, REAL w2, REAL w3){

  //REAL expoent = (aux * D * pf - aux * pf * t + 0.0833333 * (pb - pf) * (6.0 * D * D * w1 + 4.0 * D * D * D * w2 + 3.0 * D * D * D * D * w3 - t*t * (6.0 * w1 + t * (4.0 * w2 + 3.0 * t * w3)))) / aux;
  //return exp(-expoent) * (w1 + 2.0 * t * w2 + 3.0 * t * t * w3) / aux;

  REAL expoent = pf*t - pf*tb;

  return (exp(expoent) * (w1 + 2.0f * t * w2 + 3.0f * t * t * w3));
  //return sb - sf;
  //return (exp(expoent) * (sb - sf));

}



__device__ float4 GetZetaPsiQuad(Ray* ray, float4 eyepos, float raySegLength, float alphaBack, float alphaFront, float sback, float sfront, int offset,
                                 float4* dev_debugdata3, float4* dev_debugdata4, float4* dev_debugdata5, float4* dev_debugdata6,
                                 float4* dev_debugdata7, float4* dev_debugdata8, float4* dev_debugdata9){

  REAL pf = alphaFront;
  REAL pb = alphaBack;
  REAL D = raySegLength;
  REAL tf = ray->t;
  REAL tb = ray->t + D;
  
  REAL ox = eyepos.x;
  REAL dx = ray->dir.x;
  REAL oy = eyepos.y;
  REAL dy = ray->dir.y;
  REAL oz = eyepos.z;
  REAL dz = ray->dir.z;

  REAL c0 = ray->currentelem.interpolfunc1.w;
  REAL c1 = ray->currentelem.interpolfunc0.x;
  REAL c2 = ray->currentelem.interpolfunc0.y;
  REAL c3 = ray->currentelem.interpolfunc0.z;
  REAL c4 = ray->currentelem.interpolfunc0.w;
  REAL c5 = ray->currentelem.interpolfunc1.x;
  REAL c6 = ray->currentelem.interpolfunc1.y;
  REAL c7 = ray->currentelem.interpolfunc1.z;

  REAL w0, w1, w2, w3; //{w0, w1, w2, w3}
  w0 = c0 + c1*ox + c2*oy + c4*ox*oy + c3*oz + c5*ox*oz + c6*oy*oz + c7*ox*oy*oz;
  w1 = c1*dx + c2*dy + c3*dz + c4*dy*ox + c5*dz*ox + c4*dx*oy + c6*dz*oy + c7*dz*ox*oy + c5*dx*oz + c6*dy*oz + c7*dy*ox*oz + c7*dx*oy*oz;
  w2 = c4*dx*dy + c5*dx*dz + c6*dy*dz + c7*dy*dz*ox + c7*dx*dz*oy + c7*dx*dy*oz;
  w3 = c7*dx*dy*dz;


  sfront = w0 + w1 * tf + w2 * tf * tf + w3 * tf * tf * tf;
  sback = w0 + w1 * (tf + D) + w2 * pow(tf + D, 2) + w3 * pow(tf + D, 3);


  if(fabsf(sback - sfront) < EPSILON)
    return make_float4(exp(- (pf * (ray->t + D) - pf * tf)), 0, 0, 0);

  REAL zeta = exp(-(pf * tb - pf * tf));

  //16 point quadrature
  REAL psi = 0.0f;
  if(constmemory.numquadpoints == 2){
    REAL quad2[2][2] = {{0.211325, 0.5}, {0.788675, 0.5}};
    for(int i=0; i<2; i++){
      psi += (D * quad2[i][1] * Integral(quad2[i][0] * D + tf, tf + D, tf, D, pb, pf, sback, sfront, w0, w1, w2, w3));
    }
    //psi = Integral(quad1[0][0] * D, tb, tf, aux, D, pb, pf, w0, w1, w2, w3);
  }
  else if(constmemory.numquadpoints == 4){
    REAL quad4[4][2] = {{0.0694318, 0.173927},
                        {0.330009, 0.326073},
                        {0.669991, 0.326073},
                        {0.930568, 0.173927}};
    for(int i=0; i<4; i++)
      psi += (D * quad4[i][1] * Integral(quad4[i][0] * D + tf, tf + D, tf, D, pb, pf, sback, sfront, w0, w1, w2, w3));
  }
  else if(constmemory.numquadpoints == 8){
    REAL quad8[8][2] = {{0.0198551 , 0.0506143 },
                      {0.101667 , 0.111191 }, 
                      {0.237234 , 0.156853 },
                      {0.408283 , 0.181342 },
                      {0.591717 , 0.181342 },
                      {0.762766 , 0.156853 },
                      {0.898333 , 0.111191 },
                      {0.980145 , 0.0506143 }};
    for(int i=0; i<8; i++)
      psi += (D * quad8[i][1] * Integral(quad8[i][0] * D + tf, tf + D, tf, D, pb, pf, sback, sfront, w0, w1, w2, w3));
  }
  else if(constmemory.numquadpoints == 16){
    REAL quad16[16][2] = {{0.00529953 , 0.0135762 },
                          {0.0277125 , 0.0311268 },
                          {0.0671844 , 0.0475793 },
                          {0.122298 , 0.0623145 },
                          {0.191062 , 0.074798 }, 
                          {0.270992 , 0.0845783 },
                          {0.359198 , 0.0913017 }, 
                          {0.452494 , 0.0947253 },
                          {0.547506 , 0.0947253 }, 
                          {0.640802 , 0.0913017 }, 
                          {0.729008 , 0.0845783 }, 
                          {0.808938 , 0.074798 }, 
                          {0.877702 , 0.0623145 }, 
                          {0.932816 , 0.0475793 },
                          {0.972288 , 0.0311268 },
                          {0.9947 , 0.0135762 }};
    for(int i=0; i<16; i++)
      psi += (D * quad16[i][1] * Integral(quad16[i][0] * D + tf, tf + D, tf, D, pb, pf, sback, sfront, w0, w1, w2, w3));
  }

  /*
  float den = 0;
  if(constmemory.debug1 > 0)
    den = (tb*w1 - tf*w1 + tb*tb*w2 - tf*tf*w2 + tb*tb*tb*w3 - tf*tf*tf*w3);
  else
    den = ((sback) - (sfront));
  */
  //float den = (tb*w1 - tf*w1 + tb*tb*w2 - tf*tf*w2 + tb*tb*tb*w3 - tf*tf*tf*w3);
  float den = sback - sfront;
  psi = psi / den;
  dev_debugdata3[offset] = make_float4(ray->t, ray->t + D, D, 0);
  dev_debugdata4[offset] = make_float4(w0, w1, w2, w3);
  dev_debugdata5[offset] = make_float4(zeta, psi, den, 0);
  dev_debugdata6[offset] = make_float4(sfront, sback, w0 + w1 * tf + w2 * tf * tf + w3 * tf * tf * tf, w0 + w1 * (tf + D) + w2 * pow(tf + D, 2) + w3 * pow(tf + D, 3));
  dev_debugdata7[offset] = ray->eyepos;
  dev_debugdata8[offset] = make_float4(c0, c1, c2, c3);
  dev_debugdata9[offset] = make_float4(c4, c5, c6, c7);

  return make_float4(zeta, psi, 0, 0);
  
}

/**
* Calculate ZetaPsi, fetching psi gamma from texture (hexahedral mesh)
*/
__device__ float4 GetZetaPsiFetch(Ray* ray, float4 eyepos, float raySegLength, float alphaBack, float alphaFront, float sback, float sfront){
  
  float2 alphaL; // alpha * rayLength
  alphaL = raySegLength * make_float2(alphaBack, alphaFront);

  //Zeta
  float zeta = expf(-dot(alphaL, make_float2(0.5f, 0.5f)));

  //Gamma
  float2 gamma = alphaL / (alphaL + make_float2(1.0f));

  //Psi
  float psi = tex2D(texZetaPsiGamma, gamma.x, gamma.y);
  return make_float4(zeta, psi, psi, psi);

}

/**
* Find cubic roots (xyz: roots, w: number of roots)
* ax^3 + bx^2 + cx + d = 0
*/
__device__ float4 CubicRoots(float a, float b, float c, float d){

  float x1, x2, x3;
  float Q = (b * b - 3.0f * c) / 9.0f;
  float R = (2 * b * b * b - 9.0f * b * c + 27.0f * d) / 54.0f;

  //Three real roots
  if(R * R < Q * Q * Q){
    float teta = acos(R / sqrt(Q * Q * Q));
    x1 = - 2.0f * sqrt(Q) * cos(teta / 3.0f) - b / 3.0f;
    x2 = - 2.0f * sqrt(Q) * cos((teta + 2.0f * PI) / 3.0f) - b / 3.0f;
    x3 = - 2.0f * sqrt(Q) * cos((teta - 2.0f * PI) / 3.0f) - b / 3.0f;
    //x1 = fminf(x1, - 2.0f * sqrt(Q) * cos((teta + 2.0f * PI) / 3.0f) - b / 3.0f);
    //x1 = fminf(x1, - 2.0f * sqrt(Q) * cos((teta - 2.0f * PI) / 3.0f) - b / 3.0f);
    return make_float4(x1, x2, x3, 3);
  }
  //One real root
  else{
    float A;
    if(R > 0)
      A = - pow(R + sqrt(R * R - Q * Q * Q), 1.0f / 3.0f);
    else
      A = pow(-R + sqrt(R * R - Q * Q * Q), 1.0f / 3.0f);

    if(fabsf(A) > EPSILON)
      x1 = (A + Q / A) - b / 3.0f;
    else
      x1 = (A) - b / 3.0f;

    return make_float4(x1, CUDART_INF_F, CUDART_INF_F, 1);
  }
}

/**
* Find quadratic roots (xyz: roots, w: number of roots)
* ax^2 + bx + c = 0
*/
__device__ float4 QuadraticRoots(float a, float b, float c){

  float q;
  if(b < 0)
    q = -0.5 * (b - sqrt(b * b - 4 * a * c));
  else
    q = -0.5 * (b + sqrt(b * b - 4 * a * c));
  
  float x1 = q / a;
  float x2 = c / q;

  float numroots = 2.0f;
  if(x1 != x1) numroots--;
  if(x2 != x2) numroots--;

  return make_float4(x1, x2, CUDART_INF_F, numroots);
}

/**
* Find integration step (hex)
*/
inline __device__ float4 FindIntegrationStep(int* numroots, float* arrayroots, Ray* ray, float t, float cpscalar, int offset, float4* dev_debugdata0, float4* dev_debugdata1, float4* dev_debugdata2){
  

  float c0 = ray->currentelem.interpolfunc1.w;
  float c1 = ray->currentelem.interpolfunc0.x;
  float c2 = ray->currentelem.interpolfunc0.y;
  float c3 = ray->currentelem.interpolfunc0.z;
  float c4 = ray->currentelem.interpolfunc0.w;
  float c5 = ray->currentelem.interpolfunc1.x;
  float c6 = ray->currentelem.interpolfunc1.y;
  float c7 = ray->currentelem.interpolfunc1.z;
  
  float ox = ray->eyepos.x;
  float oy = ray->eyepos.y;
  float oz = ray->eyepos.z;

  //ray->dir = normalize(ray->dir);
  float dx = ray->dir.x;
  float dy = ray->dir.y;
  float dz = ray->dir.z;

  float d = c0 + c1*ox + c2*oy + c4*ox*oy + c3*oz + c5*ox*oz + c6*oy*oz + c7*ox*oy*oz - cpscalar;
  float c = c1*dx + c2*dy + c3*dz + c4*dy*ox + c5*dz*ox + c4*dx*oy + c6*dz*oy + c7*dz*ox*oy + c5*dx*oz + c6*dy*oz + c7*dy*ox*oz + c7*dx*oy*oz;
  float b = c4*dx*dy + c5*dx*dz + c6*dy*dz + c7*dy*dz*ox + c7*dx*dz*oy + c7*dx*dy*oz;
  float a = c7*dx*dy*dz;

  float4 roots;

  //ax3 + bx2 + cx + d = 0;
  //first derivate: 3ax2 + 2bx + c = 0 (find max and min)
  /*
  =======================
  Max and min
  =======================
  */
  
  //float4 pos = ray->eyepos + ray->dir * t;
  //float3 N = normalize(make_float3(c1 + c4*pos.y + c5*pos.z + c7*pos.y*pos.z, c2 + c4*pos.x + c6*pos.z + c7*pos.x*pos.z, c3 + c5*pos.x + c6*pos.y + c7*pos.x*pos.y));
  //a = N.x;
  //b = N.y;
  //c = N.z;
  
  /*
  if(fabsf(a) <= EPSILON && fabsf(b) > EPSILON){
    //Linear function
    roots.x = (- c / (2.0f * b));
    if(roots.x >= ray->t - EPSILON && roots.x <= t + EPSILON){
      arrayroots[*numroots] = roots.x;
      (*numroots) ++;
    }
  }
  else{
    //Quadratic function
    roots = QuadraticRoots(3.0f*a, 2.0f*b, c);
    if(roots.x >= ray->t - EPSILON && roots.x <= t + EPSILON){
      arrayroots[*numroots] = roots.x;
      (*numroots) ++;
    }
    if(roots.y >= ray->t - EPSILON && roots.y <= t + EPSILON){
      arrayroots[*numroots] = roots.y;
      (*numroots) ++;
    }
  }
  */

  /*
  =======================
  Cubic roots
  =======================
  */
  float inv = -1.0f;
  
  if(fabsf(a) > EPSILON && fabsf(b) > EPSILON && fabsf(c) > EPSILON && fabsf(d) > EPSILON) {
    //Cubic function
    inv = 1.0f / a;
    roots = CubicRoots(1.0f, b * inv, c * inv, d * inv);
    if(roots.x >= ray->t - EPSILON && roots.x <= t + EPSILON){
      arrayroots[*numroots] = roots.x;
      (*numroots) ++;
    }
    if(roots.y >= ray->t - EPSILON && roots.y <= t + EPSILON){
      arrayroots[*numroots] = roots.y;
      (*numroots) ++;
    }
    if(roots.z >= ray->t - EPSILON && roots.z <= t + EPSILON){
      arrayroots[*numroots] = roots.z;
      (*numroots) ++;
    }
  }
  else if(fabsf(b) > EPSILON && fabsf(c) > EPSILON && fabsf(d) > EPSILON){
    //Quadratic function
    roots = QuadraticRoots(b, c, d);
    if(roots.x >= ray->t - EPSILON && roots.x <= t + EPSILON){
      arrayroots[*numroots] = roots.x;
      (*numroots) ++;
    }
    if(roots.y >= ray->t - EPSILON && roots.y <= t + EPSILON){
      arrayroots[*numroots] = roots.y;
      (*numroots) ++;
    }
  }
  else if(fabsf(c) > EPSILON){
    //Linear function
    roots.x = (-d / c);
    if(roots.x >= ray->t - EPSILON && roots.x <= t + EPSILON){
      arrayroots[*numroots] = (-d / c);
      (*numroots) ++;
    }
  }
  
  
  /*
  if(constmemory.debug1){
    if(roots.w == 0)
      ray->acccolor = make_float4(1, 1, 1, 1);
    else if(roots.w == 1 && (roots.x >= ray->t && roots.x <= t))
      ray->acccolor = make_float4(1, 0, 0, 1);
    else if(roots.w == 2 && ((roots.x >= ray->t && roots.x <= t) || (roots.y >= ray->t && roots.y <= t)))
      ray->acccolor = make_float4(0, 1, 0, 1);
    else if(roots.w == 3 && ((roots.x >= ray->t && roots.x <= t) || (roots.y >= ray->t && roots.y <= t) || (roots.z >= ray->t && roots.z <= t)))
      ray->acccolor = make_float4(0, 0, 1, 1);
  }
  */
  /*
  if(roots.x > ray->t + EPSILON && roots.x < t - EPSILON)
    return roots.x;
  else if(roots.y > ray->t + EPSILON && roots.y < t - EPSILON)
    return roots.y;
  else if(roots.z > ray->t + EPSILON && roots.z < t - EPSILON)
    return roots.z;
  return t;
  */
  
  //Sort
  /*
  if((roots.x < roots.y && roots.x < roots.z) && (roots.y < roots.z))
    roots = make_float4(roots.x, roots.y, roots.z, roots.w);
  else if((roots.x < roots.y && roots.x < roots.z) && (roots.z < roots.y))
    roots = make_float4(roots.x, roots.z, roots.y, roots.w);
  else if((roots.y < roots.x && roots.y < roots.z) && (roots.x < roots.z))
    roots = make_float4(roots.y, roots.x, roots.z, roots.w);
  else if((roots.y < roots.x && roots.y < roots.z) && (roots.z < roots.x))
    roots = make_float4(roots.y, roots.z, roots.x, roots.w);
  else if((roots.z < roots.x && roots.z < roots.x) && (roots.x < roots.y))
    roots = make_float4(roots.z, roots.x, roots.y, roots.w);
  else if((roots.z < roots.x && roots.z < roots.x) && (roots.y < roots.x))
    roots = make_float4(roots.z, roots.y, roots.x, roots.w);
  */
  dev_debugdata0[offset] = ray->dir;
  dev_debugdata1[offset] = make_float4(inv, b *inv , c * inv, d + cpscalar);

  //return roots;


  

  //return (ray->t) + (t - ray->t) * (diffcpfront / diffbackfront);
  //return 0;
  //return ray->t + x1;

}

/**
* Bubble sort
*/
__device__ void BubbleSort(int numvalues, float* values){

  float temp;
  bool flag = true;
  for(int i=1; i < numvalues && flag; i++){
    flag = false;
    for(int j=0; j<numvalues -1 ; j++){
      if(values[j+1] < values[j]){
        temp = values[j];
        values[j] = values[j+1];
        values[j+1] = temp;
        flag = true;
      }
    }
  }
}


#else
/**
* Calculate ZetaPsi, using gaussian quadrature (tetrahedral mesh)
*/
__device__ float4 GetZetaPsiQuad(Ray* ray, float4 eyepos, float raySegLength, float alphaBack, float alphaFront, float sback, float sfront, int offset,
                                 float4* dev_debugdata3, float4* dev_debugdata4, float4* dev_debugdata5, float4* dev_debugdata6,
                                 float4* dev_debugdata7, float4* dev_debugdata8, float4* dev_debugdata9){
  float4 t1, weights1, expf_psi1, t2, weights2, expf_psi2;
  float psi = 0.0f;
  float2 alphaL; // alpha * rayLength
  alphaL = raySegLength * make_float2(alphaBack, alphaFront);

  //Zeta
  float zeta = expf(-dot(alphaL, make_float2(0.5f, 0.5f)));

  //Psi
  if(constmemory.numquadpoints == 2){
    float quad2[2][2] = {{0.211325, 0.5}, {0.788675, 0.5}};

    for(int i=0; i<2; i++){
      float expoent = - raySegLength * ((quad2[i][0]) * (alphaBack * (1.0f-quad2[i][0]) + alphaFront * quad2[i][0]));
      psi += (quad2[i][1] * expf(expoent));
    }
  }
  else if(constmemory.numquadpoints == 4){
    float quad4[4][2] = {{0.0694318, 0.173927},
                          {0.330009, 0.326073},
                          {0.669991, 0.326073},
                          {0.930568, 0.173927}};

    for(int i=0; i<4; i++){
      float expoent = - raySegLength * ((quad4[i][0]) * (alphaBack * (1.0f-quad4[i][0]) + alphaFront * quad4[i][0]));
      psi += (quad4[i][1] * expf(expoent));
    }
  }
  else if(constmemory.numquadpoints == 8){
    float quad8[8][2] = {{0.0198551 , 0.0506143 },
                        {0.101667 , 0.111191 }, 
                        {0.237234 , 0.156853 },
                        {0.408283 , 0.181342 },
                        {0.591717 , 0.181342 },
                        {0.762766 , 0.156853 },
                        {0.898333 , 0.111191 },
                        {0.980145 , 0.0506143 }};

    for(int i=0; i<8; i++){
      float expoent = - raySegLength * ((quad8[i][0]) * (alphaBack * (1.0f-quad8[i][0]) + alphaFront * quad8[i][0]));
      psi += (quad8[i][1] * expf(expoent));
    }
  }
  else if(constmemory.numquadpoints == 16){
    float quad16[16][2] = {{0.00529953 , 0.0135762 },
                          {0.0277125 , 0.0311268 },
                          {0.0671844 , 0.0475793 },
                          {0.122298 , 0.0623145 },
                          {0.191062 , 0.074798 }, 
                          {0.270992 , 0.0845783 },
                          {0.359198 , 0.0913017 }, 
                          {0.452494 , 0.0947253 },
                          {0.547506 , 0.0947253 }, 
                          {0.640802 , 0.0913017 }, 
                          {0.729008 , 0.0845783 }, 
                          {0.808938 , 0.074798 }, 
                          {0.877702 , 0.0623145 }, 
                          {0.932816 , 0.0475793 },
                          {0.972288 , 0.0311268 },
                          {0.9947 , 0.0135762 }};

    for(int i=0; i<16; i++){
      float expoent = - raySegLength * ((quad16[i][0]) * (alphaBack * (1.0f-quad16[i][0]) + alphaFront * quad16[i][0]));
      psi += (quad16[i][1] * expf(expoent));
    }
  }

  return make_float4(zeta, psi, psi, psi);
}


/**
* Calculate ZetaPsi, fetching psi gamma from texture (tetrahedral mesh)
*/
__device__ float4 GetZetaPsiFetch(Ray* ray, float4 eyepos, float raySegLength, float alphaBack, float alphaFront, float sback, float sfront){

  float2 alphaL; // alpha * rayLength
  alphaL = raySegLength * make_float2(alphaBack, alphaFront);

  //Zeta
  float zeta = expf(-dot(alphaL, make_float2(0.5f, 0.5f)));

  //Gamma
  float2 gamma = alphaL / (alphaL + make_float2(1.0f));

  //Psi
  float psi = tex2D(texZetaPsiGamma, gamma.x, gamma.y);
  return make_float4(zeta, psi, psi, psi);
}

/**
* Find scalar of the (x,y,z) point (tetrahedral mesh)
*/
inline __device__ float FindScalar(Ray* ray, float p_t){

  float4 pos = ray->eyepos + p_t * ray->dir;
  pos.w = 1.0;

  return dot(pos, ray->currentelem.interpolfunc0);
}

/**
* Find integration step (tet)
*/
inline __device__ float FindIntegrationStep(Ray* ray, float t, float diffcpfront, float diffbackfront, float isoscalar, int offset, float4* dev_debug0, float4* dev_debug1, float4* dev_debug2){
  return (ray->t) + (t - ray->t) * (diffcpfront / diffbackfront);
  //return (isoscalar - dot(ray->eyepos, ray->currentelem.interpolfunc0) - ray->currentelem.interpolfunc0.w) / (dot(ray->dir, ray->currentelem.interpolfunc0));
}

/**
* Find control point (tet)
*/
inline __device__ float FindControlPoint(Ray* ray, float backscalar, float4 cpvalues){

  float cpscalar;
  float cpnextscalar;

  if(ray->frontscalar > backscalar){
    cpscalar = cpvalues.z;
    cpnextscalar = cpvalues.w;

    if(ray->frontscalar <= cpscalar)
      cpscalar = cpnextscalar;
  }
  else{
    cpscalar = cpvalues.x;
    cpnextscalar = cpvalues.y;

    if(ray->frontscalar >= cpscalar)
      cpscalar = cpnextscalar;
  }

  return cpscalar;
}

#endif



/**
* Constant integration of the ray
*/
__device__ void IntegrateRayConst(Ray* ray, float4 eyepos, float raySegLength, float sback, float sfront){

  float4 avg = tex1D(texVolumetricColorScale, 0.5*(sback+sfront));
  //float4 avg = tex1D(texColorScale, sback);
  float zeta = expf(- raySegLength * avg.w);

  float alpha = 1 - zeta;
#ifdef CUDARC_WHITE
  float3 color = (make_float3(1) - make_float3(avg)) * alpha;
  ray->acccolor += (1 - ray->acccolor.w) * make_float4(-color.x, -color.y, -color.z, alpha);
#else
  float3 color = (make_float3(avg)) * alpha;
  ray->acccolor += (1 - ray->acccolor.w) * make_float4(color.x, color.y, color.z, alpha);
#endif

  //ray->acccolor = make_float4(1, 0, 0, 1);
}


/**
* Linear integration/trilinear of the ray
*/
__device__ void IntegrateRayLinear(Ray* ray, float4 eyepos, float raySegLength, float sback, float sfront, int offset,
                                   float4* dev_debugdata3, float4* dev_debugdata4, float4* dev_debugdata5,
                                   float4* dev_debugdata6, float4* dev_debugdata7, float4* dev_debugdata8, float4* dev_debugdata9){

  float3 color;
  float alpha;

#ifdef CUDARC_HEX
  float4 colorBack = tex1D(texVolumetricColorScale, sfront);
  float4 colorFront = tex1D(texVolumetricColorScale, sback);
#else
  float4 colorBack = tex1D(texVolumetricColorScale, sback);
  float4 colorFront = tex1D(texVolumetricColorScale, sfront);
#endif
  float4 zetapsi;

  //ray->acccolor = make_float4(colorFront.x, colorFront.y, colorFront.z, 1.0f);
  //return;

  if(constmemory.interpoltype == Quad)
    zetapsi = GetZetaPsiQuad(ray, eyepos, raySegLength, colorBack.w, colorFront.w, sback, sfront, offset,
                             dev_debugdata3, dev_debugdata4, dev_debugdata5, dev_debugdata6,
                             dev_debugdata7, dev_debugdata8, dev_debugdata9);
  else
    zetapsi = GetZetaPsiFetch(ray, eyepos, raySegLength, colorBack.w, colorFront.w, sback, sfront);

  alpha = 1.0f - zetapsi.x;
  //Finally
#ifdef CUDARC_HEX
#ifdef CUDARC_WHITE
  color = (make_float3(1)- make_float3(colorFront)) * (zetapsi.y - zetapsi.x) + (make_float3(1) - make_float3(colorBack)) * (1.0f - zetapsi.y);
  ray->acccolor += (1.0f - ray->acccolor.w) * make_float4(-color.x, -color.y, -color.z, alpha);
#else
  color = make_float3(colorFront) * (zetapsi.y - zetapsi.x) + make_float3(colorBack) * (1.0f - zetapsi.y);
  ray->acccolor += (1.0f - ray->acccolor.w) * make_float4(color.x, color.y, color.z, alpha);
#endif
#else
#ifdef CUDARC_WHITE
  color = (make_float3(1) - make_float3(colorBack)) * (zetapsi.y - zetapsi.x) + (make_float3(1)- make_float3(colorFront)) * (1.0f - zetapsi.y);
  ray->acccolor += (1 - ray->acccolor.w) * make_float4(-color.x, -color.y, -color.z, alpha);
#else
  color = (make_float3(colorBack)) * (zetapsi.y - zetapsi.x) + (make_float3(colorFront)) * (1.0f - zetapsi.y);
  ray->acccolor += (1 - ray->acccolor.w) * make_float4(color.x, color.y, color.z, alpha);
#endif
#endif
  
  //zetapsi.z = fabs(zetapsi.z);
  /*
  if(constmemory.debug1 > 0){
    if(zetapsi.y < 0)
      ray->acccolor = make_float4(1, 0, 0, 1);
    else if(zetapsi.y > 1)
      ray->acccolor = make_float4(0, 1, 0, 1);
    else
      ray->acccolor = make_float4(0, 0, 1, 1);
  }
  */
  
  /*
  else if(zetapsi.y == 0.0f)
    ray->acccolor = make_float4(1, 0, 1, 1);
  else if(zetapsi.y == 1.0f)
    ray->acccolor = make_float4(0, 1, 1, 1);
  else
    ray->acccolor = make_float4(0, 0, 1, 1);
  */
  /*
  if(sback == sfront)
    ray->acccolor = make_float4(1, 0, 0, 1);
  else
    ray->acccolor = make_float4(0, 0, 1, 1);
  */
  
}


/**
* Ray Probe box Intersection
*/
__device__ float IntersectProbeBox(Ray* ray, float* t, float* backid, float* backscalar, float3 probeboxmin, float3 probeboxmax){
  
  float4 pp;
  float tprobenear = - CUDART_INF_F;
  float tprobefar = CUDART_INF_F;
  float4 pfront = ray->eyepos + ray->t * ray->dir;
  pfront.w = 1.0f;


  pp.x = -1.0f; pp.y = 0.0f; pp.z = 0.0f; pp.w = probeboxmin.x;
  intersect_ray_probeplane(pfront, ray->dir, pp, &tprobenear, &tprobefar);
  
  pp.x = 0.0f; pp.y = -1.0f; pp.z = 0.0f; pp.w = probeboxmin.y;
  intersect_ray_probeplane(pfront, ray->dir, pp, &tprobenear, &tprobefar);

  pp.x = 0.0f; pp.y = 0.0f; pp.z = -1.0f; pp.w = probeboxmin.z;
  intersect_ray_probeplane(pfront, ray->dir, pp, &tprobenear, &tprobefar);

  pp.x = 1.0f; pp.y = 0.0f; pp.z = 0.0f; pp.w = -probeboxmax.x;
  intersect_ray_probeplane(pfront, ray->dir, pp, &tprobenear, &tprobefar);

  pp.x = 0.0f; pp.y = 1.0f; pp.z = 0.0f; pp.w = -probeboxmax.y;
  intersect_ray_probeplane(pfront, ray->dir, pp, &tprobenear, &tprobefar);

  pp.x = 0.0f; pp.y = 0.0f; pp.z = 1.0f; pp.w = -probeboxmax.z;
  intersect_ray_probeplane(pfront, ray->dir, pp, &tprobenear, &tprobefar);
  
  
  float l = *t - ray->t;

  
  //Ray leaving probe box
  if(l > tprobefar){
    *backid = 0.0f;
    if(tprobefar < 0){
      *t = ray->t;
      l = 0;
      //ray->acccolor += make_float4(0.5, 0, 0, 1);
    }
    else{
      l = tprobefar;
      *t = l + ray->t;
      *backscalar = FindScalar(ray, *t);
      //*backscalar += -(*backscalar - ray->frontscalar)*(1.0f - tprobefar/l);
      //ray->acccolor += make_float4(0, 0.5, 0, 1);
    }
  }
  
  
  //Ray entering probe box
  if(tprobenear > 0){
    if(tprobenear >= l){
      *t = ray->t;
      l = 0;
      //ray->acccolor += make_float4(0, 0, 0.5, 1);
    }
    else{
      l -= tprobenear;
      ray->t = *t - l;
      ray->frontscalar = FindScalar(ray, *t);
      //ray->frontscalar += (*backscalar - ray->frontscalar) * tprobenear / l;
      //ray->acccolor += make_float4(0.5, 0.5, 0.5, 1);
    }
  }
  
  //*t += ray->t;
  *t = fmaxf(*t, ray->t);
  //l = fmaxf(l, 0);
  //*t = ray->t + l;

  //ray->acccolor = make_float4(*t, *t, *t, 1.0f);
  //*t += ray->t;
  
}

/**
* Ray Plane Intersection (Plucker)
*/
#if !defined(CUDARC_HEX) && defined(CUDARC_PLUCKER)
__device__ float Intersect(Ray* ray, float* t){
  float4 v0 = tex1Dfetch(texNode0, ray->frontid);
  float4 v1 = tex1Dfetch(texNode1, ray->frontid);
  float4 v2 = tex1Dfetch(texNode2, ray->frontid);
  float4 v3 = tex1Dfetch(texNode3, ray->frontid);

  float4 dircrosseyepos = cross(ray->dir, ray->eyepos);

  float4 v02 = (v0 - v2);
  float4 q02 = cross(v02, v0);
  float ps02 = permuted_inner_produtct(v02, q02, ray->dir, dircrosseyepos);

  float4 v32 = (v3 - v2);
  float4 q32 = cross(v32, v2);
  float ps32 = permuted_inner_produtct(v32, q32, ray->dir, dircrosseyepos);

  float4 v03 = (v0 - v3);
  float4 q03 = cross(v03, v0);
  float ps03 = permuted_inner_produtct(v03, q03, ray->dir, dircrosseyepos);

  float4 v13 = (v1 - v3);
  float4 q13 = cross(v13, v1);
  float ps13 = permuted_inner_produtct(v13, q13, ray->dir, dircrosseyepos);

  float4 v01 = (v0 - v1);
  float4 q01 = cross(v01, v0);
  float ps01 = permuted_inner_produtct(v01, q01, ray->dir, dircrosseyepos);

  float4 v21 = (v2 - v1);
  float4 q21 = cross(v21, v1);
  float ps21 = permuted_inner_produtct(v21, q21, ray->dir, dircrosseyepos);

  float round = 0.0f;
  float4 point;
  //Plucker tests
  if(ray->frontface == 0.0f){

    //Test against faces 1, 2, 3
    if((-ps32 <= 0 && -ps03 <= 0 && ps02 <= 0)){
      float3 u = make_float3(-ps32, -ps03, ps02) / (-ps32 -ps03 + ps02);
      if(constmemory.debug0) ray->acccolor = make_float4(0, 1, 0, 1);
      round = ray->currentelem.adj0.y;
      point = u.x * v0 +  u.y * v2 + u.z * v3;
      point.w = 1.0f;
    }
    else{
      if((-ps13 <= 0 && -ps01 <= 0 && ps03 <= 0)){
        float3 u = make_float3(-ps13, -ps01, ps03) / (-ps13 -ps01 + ps03);
        if(constmemory.debug0) ray->acccolor = make_float4(0, 0, 1, 1);
        round = ray->currentelem.adj0.z;
        point = u.x * v0 +  u.y * v3 + u.z * v1;
        point.w = 1.0f;
      }
      else{
        if((-ps21 <= 0 && -ps02 <= 0 && ps01 <= 0)){
          float3 u = make_float3(-ps21, -ps02, ps01) / (-ps21 -ps02 + ps01);
          if(constmemory.debug0) ray->acccolor = make_float4(1, 0, 1, 1);
          round = ray->currentelem.adj0.w;
          point = u.x * v0 +  u.y * v1 + u.z * v2;
          point.w = 1.0f;
        }
      } 
    }
  }
  else if(ray->frontface == 1.0f){
    //Test against faces 0, 2, 3
    if((ps13 <= 0 && ps32 <= 0 && ps21 <= 0)){
      float3 u = make_float3(ps13, ps32, ps21) / (ps13 + ps32 + ps21);
      if(constmemory.debug0) ray->acccolor = make_float4(1, 0, 0, 1);
      round = ray->currentelem.adj0.x;
      point = u.x * v2 +  u.y * v1 + u.z * v3;
      point.w = 1.0f;
    }
    else{
      if((-ps13 <= 0 && -ps01 <= 0 && ps03 <= 0)){
        float3 u = make_float3(-ps13, -ps01, ps03) / (-ps13 -ps01 + ps03);
        if(constmemory.debug0) ray->acccolor = make_float4(0, 0, 1, 1);
        round = ray->currentelem.adj0.z;
        point = u.x * v0 +  u.y * v3 + u.z * v1;
        point.w = 1.0f;
      }
      else{
        if((-ps21 <= 0 && -ps02 <= 0 && ps01 <= 0)){
          float3 u = make_float3(-ps21, -ps02, ps01) / (-ps21 -ps02 + ps01);
          if(constmemory.debug0) ray->acccolor = make_float4(1, 0, 1, 1);
          round = ray->currentelem.adj0.w;
          point = u.x * v0 +  u.y * v1 + u.z * v2;
          point.w = 1.0f;
        }
      } 
    }
  }
  else if(ray->frontface == 2.0f){
    //Test against faces 0, 1, 3
    if((ps13 <= 0 && ps32 <= 0 && ps21 <= 0)){
      float3 u = make_float3(ps13, ps32, ps21) / (ps13 + ps32 + ps21);
      if(constmemory.debug0) ray->acccolor = make_float4(1, 0, 0, 1);
      round = ray->currentelem.adj0.x;
      point = u.x * v2 +  u.y * v1 + u.z * v3;
      point.w = 1.0f;
    }
    else{
      if((-ps32 <= 0 && -ps03 <= 0 && ps02<= 0)){
        float3 u = make_float3(-ps32, -ps03, ps02) / (-ps32 -ps03 + ps02);
        if(constmemory.debug0) ray->acccolor = make_float4(0, 1, 0, 1);
        round = ray->currentelem.adj0.y;
        point = u.x * v0 +  u.y * v2 + u.z * v3;
        point.w = 1.0f;
      }
      else{
        if((-ps21 <= 0 && -ps02 <= 0 && ps01 <= 0)){
          float3 u = make_float3(-ps21, -ps02, ps01) / (-ps21 -ps02 + ps01);
          if(constmemory.debug0) ray->acccolor = make_float4(1, 0, 1, 1);
          round = ray->currentelem.adj0.w;
          point = u.x * v0 +  u.y * v1 + u.z * v2;
          point.w = 1.0f;
        }
      } 
    }
  }
  else if(ray->frontface == 3.0f){
    //Test against faces 0, 1, 2
    if((ps13 <= 0 && ps32 <= 0 && ps21 <= 0)){
      float3 u = make_float3(ps13, ps32, ps21) / (ps13 + ps32 + ps21);
      if(constmemory.debug0) ray->acccolor = make_float4(1, 0, 0, 1);
      round = ray->currentelem.adj0.x;
      point = u.x * v2 +  u.y * v1 + u.z * v3;
      point.w = 1.0f;
    }
    else{
      if((-ps32 <= 0 && -ps03 <= 0 && ps02 <= 0)){
        float3 u = make_float3(-ps32, -ps03, ps02) / (-ps32 -ps03 + ps02);
        if(constmemory.debug0) ray->acccolor = make_float4(0, 1, 0, 1);
        round = ray->currentelem.adj0.y;
        point = u.x * v0 +  u.y * v2 + u.z * v3;
        point.w = 1.0f;
      }
      else{
        if((-ps13 <= 0 && -ps01 <= 0 && ps03 <= 0)){
          float3 u = make_float3(-ps13, -ps01, ps03) / (-ps13 -ps01 + ps03);
          if(constmemory.debug0) ray->acccolor = make_float4(0, 0, 1, 1);
          round = ray->currentelem.adj0.z;
          point = u.x * v0 +  u.y * v3 + u.z * v1;
          point.w = 1.0f;
        }
      } 
    }
  }


  *t = length(point - ray->eyepos);
  return round;
}
#endif

#if !defined(CUDARC_PLUCKER)
/**
* Ray Plane Intersection
*/
__device__ float Intersect(Ray* ray, float* t){

  float roundid;
  float tintersect;

  /*Face 0*/
  //if(ray->frontface != 0){
    //Triangle 0
    tintersect = intersect_ray_plane(ray->eyepos, ray->dir, tex1Dfetch(texFace0Eq, ray->frontid));
    if(tintersect < *t){
      *t = tintersect;
      if(constmemory.debug0 > 0) ray->acccolor = make_float4(1, 0, 0, 1);
      roundid = ray->currentelem.adj0.x;
    }
  //}

  /*Face 1*/
  //if(ray->frontface != 1){
    //Triangle 0
    tintersect = intersect_ray_plane(ray->eyepos, ray->dir, tex1Dfetch(texFace1Eq, ray->frontid));
    if(tintersect < *t){
      *t = tintersect;
      if(constmemory.debug0 > 0) ray->acccolor = make_float4(0, 1, 0, 1);
      roundid = ray->currentelem.adj0.y;
    }
  //}

  /*Face 2*/
  //if(ray->frontface != 2){
    //Triangle 0
    tintersect = intersect_ray_plane(ray->eyepos, ray->dir, tex1Dfetch(texFace2Eq, ray->frontid));
    if(tintersect < *t){
      *t = tintersect;
      if(constmemory.debug0 > 0) ray->acccolor = make_float4(0, 0, 1, 1);
      roundid = ray->currentelem.adj0.z;
    }
  //}

  /*Face 3*/
  //if(ray->frontface != 3){
    //Triangle 0
    tintersect = intersect_ray_plane(ray->eyepos, ray->dir, tex1Dfetch(texFace3Eq, ray->frontid));
    if(tintersect < *t){
      *t = tintersect;
      if(constmemory.debug0 > 0) ray->acccolor = make_float4(1, 1, 0, 1);
#ifdef CUDARC_HEX
      roundid = ray->currentelem.adj1.x;
#else
      roundid = ray->currentelem.adj0.w;
#endif
    }
  //}

#ifdef CUDARC_HEX
  /*Face 4*/
  //if(ray->frontface != 4){
    //Triangle 0
    tintersect = intersect_ray_plane(ray->eyepos, ray->dir, tex1Dfetch(texFace4Eq, ray->frontid));
    if(tintersect < *t){
      *t = tintersect;
      if(constmemory.debug0 > 0) ray->acccolor = make_float4(1, 0, 1, 1);
      roundid = ray->currentelem.adj1.y;
    }
  //}

  /*Face 5*/
  //if(ray->frontface != 5){
    //Triangle 0
    tintersect = intersect_ray_plane(ray->eyepos, ray->dir, tex1Dfetch(texFace5Eq, ray->frontid));
    if(tintersect < *t){
      *t = tintersect;
      if(constmemory.debug0 > 0) ray->acccolor = make_float4(0, 1, 1, 1);
      roundid = ray->currentelem.adj1.z;
    }
  //}
#endif

  return roundid;
}
#endif

/**
* Ray Bilinear Patch Intersection
*/
#if defined(CUDARC_HEX) && defined(CUDARC_BILINEAR)
__device__ float Intersect(Ray* ray, float* t){
  
  float round;

  float4 v0 = tex1Dfetch(texNode0, ray->frontid);
  float4 v1 = tex1Dfetch(texNode1, ray->frontid);
  float4 v2 = tex1Dfetch(texNode2, ray->frontid);
  float4 v3 = tex1Dfetch(texNode3, ray->frontid);
  float4 v4 = tex1Dfetch(texNode4, ray->frontid);
  float4 v5 = tex1Dfetch(texNode5, ray->frontid);
  float4 v6 = tex1Dfetch(texNode6, ray->frontid);
  float4 v7 = tex1Dfetch(texNode7, ray->frontid);


  //Ray Bilinear patch intersection
  float2 t0 = make_float2(CUDART_INF_F, 0);
  float2 t1 = make_float2(CUDART_INF_F, 0);
  float2 t2 = make_float2(CUDART_INF_F, 0);
  float2 t3 = make_float2(CUDART_INF_F, 0);
  float2 t4 = make_float2(CUDART_INF_F, 0);
  float2 t5 = make_float2(CUDART_INF_F, 0);


  ///res
  /*
  t0.x = IntersectBilinear(ray, v0, v1, v2, v3, ray->frontface == 0 ? 1 : 0).x;
  t1.x = fminf(t, IntersectBilinear(ray, v4, v5, v6, v7, ray->frontface == 1 ? 1 : 0).x);
  t2.x = fminf(t, IntersectBilinear(ray, v1, v3, v5, v7, ray->frontface == 2 ? 1 : 0).x);
  t3.x = fminf(t, IntersectBilinear(ray, v0, v2, v4, v6, ray->frontface == 3 ? 1 : 0).x);
  t4.x = fminf(t, IntersectBilinear(ray, v2, v3, v6, v7, ray->frontface == 5 ? 1 : 0).x);
  t5.x = fminf(t, IntersectBilinear(ray, v0, v1, v4, v5, ray->frontface == 4 ? 1 : 0).x);
  */

  //fem
  t0.x = IntersectBilinear(ray, v0, v1, v3, v2, ray->frontface == 1 ? 1 : 0).x;
  t1.x = fminf(*t, IntersectBilinear(ray, v5, v4, v6, v7, ray->frontface == 0 ? 1 : 0).x);
  t2.x = fminf(*t, IntersectBilinear(ray, v1, v2, v5, v6, ray->frontface == 2 ? 1 : 0).x);
  t3.x = fminf(*t, IntersectBilinear(ray, v0, v3, v4, v7, ray->frontface == 3 ? 1 : 0).x);
  t4.x = fminf(*t, IntersectBilinear(ray, v2, v3, v6, v7, ray->frontface == 5 ? 1 : 0).x);
  t5.x = fminf(*t, IntersectBilinear(ray, v0, v1, v4, v5, ray->frontface == 4 ? 1 : 0).x);


  if(t0.x < t1.x && t0.x < t2.x && t0.x < t3.x && t0.x < t4.x && t0.x < t5.x){
    *t = t0.x;
    //round = hexAdj1.y; //fem
    round = ray->currentelem.adj0.x; //res
    //round = 0;
    if(constmemory.debug0) ray->acccolor = make_float4(1, 0, 0, 1);
  }
  if(t1.x < t0.x && t1.x < t2.x && t1.x < t3.x && t1.x < t4.x && t1.x < t5.x){
    *t = t1.x;
    //round = hexAdj1.x; //fem
    round = ray->currentelem.adj0.y; //res
    //round = 0;
    if(constmemory.debug0) ray->acccolor = make_float4(0, 1, 0, 1);
  }
  if(t2.x < t0.x && t2.x < t1.x && t2.x < t3.x && t2.x < t4.x && t2.x < t5.x){
    *t = t2.x;
    //round = hexAdj1.z; //fem
    round = ray->currentelem.adj0.z; //res
    //round = 0;
    if(constmemory.debug0) ray->acccolor = make_float4(0, 1, 1, 1);
  }
  if(t3.x < t0.x && t3.x < t1.x && t3.x < t2.x && t3.x < t4.x && t3.x < t5.x){
    *t = t3.x;
    //round = hexAdj2.x; //fem
    round = ray->currentelem.adj1.x; //res
    //round = 0;
    if(constmemory.debug0) ray->acccolor = make_float4(0, 0, 1, 1);
  }
  if(t4.x < t0.x && t4.x < t1.x && t4.x < t2.x && t4.x < t3.x && t4.x < t5.x){
    *t = t4.x;
    //round = hexAdj2.z; //fem
    round = ray->currentelem.adj1.x; //res
    //round = 0;
    if(constmemory.debug0) ray->acccolor = make_float4(1, 1, 0, 1);
  }
  if(t5.x < t0.x && t5.x < t1.x && t5.x < t2.x && t5.x < t3.x && t5.x < t4.x){
    *t = t5.x;
    //round = hexAdj2.y; //fem
    round = ray->currentelem.adj1.y; //res
    //round = 0;
    if(constmemory.debug0) ray->acccolor = make_float4(1, 0, 1, 1);
  }

  if(constmemory.debug0) return;

  return round;
}
#endif


/**
* Initialize function, calculate the starting position of the ray on the mesh
*/
__device__ Ray Initialize(int x, int y, int offset, float4 eyePos){

  float4 tetraInfo = tex2D(texIntersect, x, y);
  float4 dir = make_float4(tetraInfo.x, tetraInfo.y, tetraInfo.z, 0);
  int tid = floor(tetraInfo.w + 0.5f);

  Ray ray;
  ray.t = 0.0f;
  ray.dir = normalize(dir);
  ray.eyepos = eyePos;
#ifdef CUDARC_WITH_FACEID
#ifdef CUDARC_HEX
  ray.frontid = tid / 6; 
  ray.frontface = tid % 6; 
#else
  ray.frontid = tid * 0.25f;  
  ray.frontface = tid % 4;  
#endif
#else
  ray.frontid = tid;
#endif

#ifdef CUDARC_WHITE
  ray.acccolor = make_float4(1, 1, 1, 0);
#else
  ray.acccolor = make_float4(0);
#endif

  
  ray.currentelem.interpolfunc0 = tex1Dfetch(texInterpolFunc0, ray.frontid);
#ifdef CUDARC_HEX
  ray.currentelem.interpolfunc1 = tex1Dfetch(texInterpolFunc1, ray.frontid);
#endif


#ifdef CUDARC_BILINEAR
  float4 v0 = tex1Dfetch(texNode0, ray.frontid);
  float4 v1 = tex1Dfetch(texNode1, ray.frontid);
  float4 v2 = tex1Dfetch(texNode2, ray.frontid);
  float4 v3 = tex1Dfetch(texNode3, ray.frontid);
  float4 v4 = tex1Dfetch(texNode4, ray.frontid);
  float4 v5 = tex1Dfetch(texNode5, ray.frontid);
  float4 v6 = tex1Dfetch(texNode6, ray.frontid);
  float4 v7 = tex1Dfetch(texNode7, ray.frontid);

  if(ray.frontface == 0)
    ray.t = IntersectBilinear(&ray, v5, v4, v6, v7, false).x; //fem
    //ray.t = IntersectBilinear(&ray, v0, v1, v2, v3, false).x; //res
  else if(ray.frontface == 1)
    ray.t = IntersectBilinear(&ray, v0, v1, v3, v2, false).x; //fem
    //ray.t = IntersectBilinear(&ray, v4, v5, v6, v7, false).x; //res
  else if(ray.frontface == 2)
    ray.t = IntersectBilinear(&ray, v1, v2, v5, v6, false).x; //fem
    //ray.t = IntersectBilinear(&ray, v1, v3, v5, v7, false).x; //res
  else if(ray.frontface == 3)
    ray.t = IntersectBilinear(&ray, v0, v3, v4, v7, false).x; //fem
    //ray.t = IntersectBilinear(&ray, v0, v2, v4, v6, false).x; //res
  else if(ray.frontface == 4)
    ray.t = IntersectBilinear(&ray, v0, v1, v4, v5, false).x; //fem
    //ray.t = IntersectBilinear(&ray, v0, v1, v4, v5, false).x; //res
  else if(ray.frontface == 5)
    ray.t = IntersectBilinear(&ray, v2, v3, v6, v7, false).x; //fem
    //ray.t = IntersectBilinear(&ray, v2, v3, v6, v7, false).x; //res

  if(ray.t == CUDART_INF_F)
    ray.acccolor = make_float4(0,0,0,1);

  //ray.t = length(dir);

#else
   ray.t = length(dir);
#endif

  ray.frontscalar = FindScalar(&ray, ray.t);
  /*
  if(ray.t == CUDART_INF_F)
    ray.acccolor = make_float4(1, 0, 0, 1);
  else
    ray.acccolor = make_float4(0, 0, 1, 1);
  */

  return ray;

}

/**
* Volumetric traverse the ray through the mesh
*/
__device__ void Traverse(int x, int y, int offset, Ray* ray, float3 probeboxmin, float3 probeboxmax, 
                         float4* dev_debugdata0, float4* dev_debugdata1, float4* dev_debugdata2,
                         float4* dev_debugdata3, float4* dev_debugdata4, float4* dev_debugdata5,
                         float4* dev_debugdata6, float4* dev_debugdata7, float4* dev_debugdata8,
                         float4* dev_debugdata9){

  float4 planeEq;
  float sameDirection;
  float t = CUDART_INF_F;
  float backid = 0;
  float backfaceid = 0;
  float backscalar;
  float round = 0;

  ray->currentelem.adj0 = tex1Dfetch(texAdj0, ray->frontid);
#ifdef CUDARC_HEX
  ray->currentelem.adj1 = tex1Dfetch(texAdj1, ray->frontid);
#endif

  
//#ifdef CUDARC_HEX
//  float4 v0, v1, v2, v3, v4, v5, v6, v7;
//  float4 ray = cross(ray->dir, ray->eyepos);
//  /*
//  float4 v0 = tex1Dfetch(texNode0, ray->frontid);
//  float4 v1 = tex1Dfetch(texNode1, ray->frontid);
//  float4 v2 = tex1Dfetch(texNode2, ray->frontid);
//  float4 v3 = tex1Dfetch(texNode3, ray->frontid);
//  float4 v4 = tex1Dfetch(texNode4, ray->frontid);
//  float4 v5 = tex1Dfetch(texNode5, ray->frontid);
//  float4 v6 = tex1Dfetch(texNode6, ray->frontid);
//  float4 v7 = tex1Dfetch(texNode7, ray->frontid);
//  float4 ray = cross(ray->dir, ray->eyepos);
//  float4 point;
//  
//  float4 v02 = (v0 - v2);
//  float4 q02 = cross(v02, v0);
//  float ps02 = permuted_inner_produtct(v02, q02, ray->dir, ray);
//
//  float4 v32 = (v3 - v2);
//  float4 q32 = cross(v32, v2);
//  float ps32 = permuted_inner_produtct(v32, q32, ray->dir, ray);
//
//  float4 v13 = (v1 - v3);
//  float4 q13 = cross(v13, v1);
//  float ps13 = permuted_inner_produtct(v13, q13, ray->dir, ray);
//
//  float4 v01 = (v0 - v1);
//  float4 q01 = cross(v01, v0);
//  float ps01 = permuted_inner_produtct(v01, q01, ray->dir, ray);
//
//  float4 v21 = (v2 - v1);
//  float4 q21 = cross(v21, v1);
//  float ps21 = permuted_inner_produtct(v21, q21, ray->dir, ray);
//
//  float4 v76 = (v7 - v6);
//  float4 q76 = cross(v76, v6);
//  float ps76 = permuted_inner_produtct(v76, q76, ray->dir, ray);
//
//  float4 v57 = (v5 - v7);
//  float4 q57 = cross(v57, v7);
//  float ps57 = permuted_inner_produtct(v57, q57, ray->dir, ray);
//
//  float4 v46 = (v4 - v6);
//  float4 q46 = cross(v46, v6);
//  float ps46 = permuted_inner_produtct(v46, q46, ray->dir, ray);
//
//  float4 v45 = (v4 - v5);
//  float4 q45 = cross(v45, v5);
//  float ps45 = permuted_inner_produtct(v45, q45, ray->dir, ray);
//
//  float4 v26 = (v2 - v6);
//  float4 q26 = cross(v26, v6);
//  float ps26 = permuted_inner_produtct(v26, q26, ray->dir, ray);
//
//  float4 v37 = (v3 - v7);
//  float4 q37 = cross(v37, v7);
//  float ps37 = permuted_inner_produtct(v37, q37, ray->dir, ray);
//
//  float4 v04 = (v0 - v4);
//  float4 q04 = cross(v04, v4);
//  float ps04 = permuted_inner_produtct(v04, q04, ray->dir, ray);
//
//  float4 v15 = (v1 - v5);
//  float4 q15 = cross(v15, v5);
//  float ps15 = permuted_inner_produtct(v15, q15, ray->dir, ray);
//  */
//#else
//#ifdef CUDARC_PLUCKER
//  float4 v0 = tex1Dfetch(texNode0, ray->frontid);
//  float4 v1 = tex1Dfetch(texNode1, ray->frontid);
//  float4 v2 = tex1Dfetch(texNode2, ray->frontid);
//  float4 v3 = tex1Dfetch(texNode3, ray->frontid);
//  float4 ray = cross(ray->dir, ray->eyepos);
//  float4 point;
//
//  float4 v02 = (v0 - v2);
//  float4 q02 = cross(v02, v0);
//  float ps02 = permuted_inner_produtct(v02, q02, ray->dir, ray);
//
//  float4 v32 = (v3 - v2);
//  float4 q32 = cross(v32, v2);
//  float ps32 = permuted_inner_produtct(v32, q32, ray->dir, ray);
//
//  float4 v03 = (v0 - v3);
//  float4 q03 = cross(v03, v0);
//  float ps03 = permuted_inner_produtct(v03, q03, ray->dir, ray);
//
//  float4 v13 = (v1 - v3);
//  float4 q13 = cross(v13, v1);
//  float ps13 = permuted_inner_produtct(v13, q13, ray->dir, ray);
//
//  float4 v01 = (v0 - v1);
//  float4 q01 = cross(v01, v0);
//  float ps01 = permuted_inner_produtct(v01, q01, ray->dir, ray);
//
//  float4 v21 = (v2 - v1);
//  float4 q21 = cross(v21, v1);
//  float ps21 = permuted_inner_produtct(v21, q21, ray->dir, ray);
//#endif
//#endif
//  


  int aux = 0;
  while((constmemory.numtraverses > 0 && constmemory.debug0) || (ray->frontid > 0 && ray->acccolor.w < 0.99)){
    
    if((constmemory.numtraverses > 0 && aux >= constmemory.numtraverses) || aux >= CUDARC_MAX_ITERATIONS)
      break;

    aux++;

    
    ray->dir.w = 0;
    ray->eyepos.w = 1;

    round = Intersect(ray, &t);

    int rounded = floor(round + 0.5f);
#ifdef CUDARC_WITH_FACEID
#ifdef CUDARC_HEX
    backid = rounded / 6; 
    backfaceid = rounded % 6; 
#else
    backid = rounded * 0.25f; 
    backfaceid = rounded % 4; 
#endif
#else
    backid = rounded;
#endif

    //return;

    t = fmaxf(t, ray->t);
    backscalar = FindScalar(ray, t);


    if(constmemory.interpoltype == Step){
      float frontt = ray->t;
      float step = (t - frontt) / constmemory.numsteps;
      float frontscalar = ray->frontscalar;
      backscalar = FindScalar(ray, frontt + step);
      for(int i=0; i < constmemory.numsteps; i++){
        IntegrateRayConst(ray, ray->eyepos, step / constmemory.maxedge, backscalar, frontscalar);
        frontt += (step); 

        frontscalar = backscalar;
        backscalar = FindScalar(ray, frontt);

        if(ray->acccolor.w > 0.99)
          return;
      }
    }
    else{

      
      
#ifdef CUDARC_HEX
      
      
      float currentcp = 0;
      int numroots = 0;
      int numvalidroots = 0;
      float arrayroots[CUDARC_MAX_CONTROLPOINTS * 3]; //TODO: use shared memory. Array uses global memory
      float cpscalar = tex1D(texVolumetricControlPoints, currentcp).x;
      currentcp++;

      
      //cpscalar = 0.5f;
      while(cpscalar < 1.0f && (constmemory.nummaxcp == 0 || currentcp < constmemory.nummaxcp)){
        FindIntegrationStep(&numroots, arrayroots, ray, t, cpscalar, offset, dev_debugdata0, dev_debugdata1, dev_debugdata2);
        cpscalar = tex1D(texVolumetricControlPoints, currentcp).x;
        currentcp++;
        //cpscalar = 3.0f;
      }
      currentcp--;

      
      if(numroots > 0){
        BubbleSort(numroots, arrayroots);

        //Integrate ray between ray->t and first root
        ray->frontscalar = FindScalar(ray, ray->t);
        float diff = arrayroots[0] - ray->t;
        backscalar = FindScalar(ray, ray->t + diff / constmemory.maxedge);
        
        if(constmemory.volumetric){
          if(constmemory.interpoltype != Const)
            IntegrateRayLinear(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar, offset,
                               dev_debugdata3, dev_debugdata4, dev_debugdata5, dev_debugdata6,
                               dev_debugdata7, dev_debugdata8, dev_debugdata9);
          else
            IntegrateRayConst(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar);
        }
                           

        
        //Integrate ray between root i and root i+1
        for(int i=0; i<numroots-1; i++){
          ray->t = arrayroots[i];
          ray->frontscalar = FindScalar(ray, ray->t);
          diff = arrayroots[i+1] - ray->t;
          backscalar = FindScalar(ray, ray->t + diff / constmemory.maxedge);

          
          if(constmemory.volumetric){
            if(constmemory.interpoltype != Const)
              IntegrateRayLinear(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar, offset,
                                 dev_debugdata3, dev_debugdata4, dev_debugdata5, dev_debugdata6,
                                 dev_debugdata7, dev_debugdata8, dev_debugdata9);
            else
              IntegrateRayConst(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar);
          }
          
          //Isosurface
          if(constmemory.isosurface > 0){
            //ray->acccolor = make_float4(0);
            float c0 = ray->currentelem.interpolfunc1.w;
            float c1 = ray->currentelem.interpolfunc0.x;
            float c2 = ray->currentelem.interpolfunc0.y;
            float c3 = ray->currentelem.interpolfunc0.z;
            float c4 = ray->currentelem.interpolfunc0.w;
            float c5 = ray->currentelem.interpolfunc1.x;
            float c6 = ray->currentelem.interpolfunc1.y;
            float c7 = ray->currentelem.interpolfunc1.z;

            float4 pos = ray->eyepos + ray->dir * ray->t;

            float3 N = normalize(make_float3(c1 + c4*pos.y + c5*pos.z + c7*pos.y*pos.z, c2 + c4*pos.x + c6*pos.z + c7*pos.x*pos.z, c3 + c5*pos.x + c6*pos.y + c7*pos.x*pos.y));

            float4 color = tex1D(texIsoColorScale, backscalar);
            //float4 color = make_float4(1, 0, 0, 0.3f);
            float3 L = make_float3(ray->dir);
            color.x *= abs(dot(N, L));
            color.y *= abs(dot(N, L));
            color.z *= abs(dot(N, L));


            color.x *= (color.w);
            color.y *= (color.w);
            color.z *= (color.w);
            ray->acccolor += (1.0f - ray->acccolor.w) * color;

            //dev_debugdata3[offset] = make_float4(ray->t, intstep, t, FindScalar(ray, intstep));
            //ray->acccolor = make_float4(0, 1, 0, 1);
            //return;

            //ray->acccolor = make_float4(1, 0, 0, 1);
            if(ray->acccolor.w > 0.99) return;

          }
                             
        }
        

        //Integrate ray between last root and t

        ray->t = arrayroots[numroots-1];
        ray->frontscalar = FindScalar(ray, ray->t);
        diff = t - ray->t;
        backscalar = FindScalar(ray, ray->t + diff / constmemory.maxedge);
        
        if(constmemory.volumetric){
          if(constmemory.interpoltype != Const)
            IntegrateRayLinear(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar, offset,
                                dev_debugdata3, dev_debugdata4, dev_debugdata5, dev_debugdata6,
                                dev_debugdata7, dev_debugdata8, dev_debugdata9);
          else
            IntegrateRayConst(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar);
        }

        //Isosurface
        if(constmemory.isosurface > 0){
          //ray->acccolor = make_float4(0);
          float c0 = ray->currentelem.interpolfunc1.w;
          float c1 = ray->currentelem.interpolfunc0.x;
          float c2 = ray->currentelem.interpolfunc0.y;
          float c3 = ray->currentelem.interpolfunc0.z;
          float c4 = ray->currentelem.interpolfunc0.w;
          float c5 = ray->currentelem.interpolfunc1.x;
          float c6 = ray->currentelem.interpolfunc1.y;
          float c7 = ray->currentelem.interpolfunc1.z;

          float4 pos = ray->eyepos + ray->dir * ray->t;


          float3 N = normalize(make_float3(c1 + c4*pos.y + c5*pos.z + c7*pos.y*pos.z, c2 + c4*pos.x + c6*pos.z + c7*pos.x*pos.z, c3 + c5*pos.x + c6*pos.y + c7*pos.x*pos.y));

          float4 color = tex1D(texIsoColorScale, backscalar);
          //float4 color = make_float4(1, 0, 0, 0.3f);
          float3 L = make_float3(ray->dir);
          color.x *= abs(dot(N, L));
          color.y *= abs(dot(N, L));
          color.z *= abs(dot(N, L));


          color.x *= (color.w);
          color.y *= (color.w);
          color.z *= (color.w);
          ray->acccolor += (1.0f - ray->acccolor.w) * color;

          //dev_debugdata3[offset] = make_float4(ray->t, intstep, t, FindScalar(ray, intstep));
          //ray->acccolor = make_float4(0, 1, 0, 1);
          //return;

          //ray->acccolor = make_float4(0, 1, 0, 1);
          if(ray->acccolor.w > 0.99) return;

        }
                            
                            
      }
      else{
        
        //if(constmemory.debug2 > 0){
        ray->frontscalar = FindScalar(ray, ray->t);
        float diff = t - ray->t;
        backscalar = FindScalar(ray, ray->t + diff / constmemory.maxedge);
        if(constmemory.volumetric){
          if(constmemory.interpoltype != Const)
            IntegrateRayLinear(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar, offset,
                              dev_debugdata3, dev_debugdata4, dev_debugdata5, dev_debugdata6,
                              dev_debugdata7, dev_debugdata8, dev_debugdata9);
          else
            IntegrateRayConst(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar);
        }
        //}
                       
      }
      
      //return;
      dev_debugdata2[offset] = make_float4(0, 0, 0, numroots);

      /*
      currentcp = 0;

      float maxcp;
      if(constmemory.debug1 > 0)
        maxcp = 3;
      else
        maxcp = 1;

      while(currentcp < maxcp){
        float intstep = ray->t;

        int countvalidroots = 0;

        
        float4 roots;
        if(currentcp == 0)
          roots = FindIntegrationStep(&numroots, arrayroots, ray, t, 0.35, offset, dev_debugdata0, dev_debugdata1, dev_debugdata2);
        else if(currentcp == 1)
          roots = FindIntegrationStep(&numroots, arrayroots, ray, t, 0.45, offset, dev_debugdata0, dev_debugdata1, dev_debugdata2);
        else if(currentcp == 2)
          roots = FindIntegrationStep(&numroots, arrayroots, ray, t, 0.75, offset, dev_debugdata0, dev_debugdata1, dev_debugdata2);
        else if(currentcp == 3)
          roots = FindIntegrationStep(&numroots, arrayroots, ray, t, 0.2, offset, dev_debugdata0, dev_debugdata1, dev_debugdata2);
        


        if(roots.w > 0){
          for(int i=0; i<roots.w; i++){
            float newstep;
            if(i==0){
              newstep = roots.x;
            }
            else if(i==1){
              newstep = roots.y;
            }
            else if(i == 2){
              newstep = roots.z;
            }
            
            if(newstep >= ray->t - EPSILON && newstep <= t + EPSILON){
              countvalidroots++;
              intstep = newstep;
              float diff = intstep - ray->t;

              
              //Integrate between front face and first root
              //ray->frontscalar = FindScalar(ray, ray->t);
              //backscalar = FindScalar(ray, intstep);
              ray->frontscalar = FindScalar(ray, ray->t);
              backscalar = FindScalar(ray, ray->t + diff / constmemory.maxedge);
              
              if(constmemory.volumetric){
                if(constmemory.interpoltype == Const)
                  IntegrateRayConst(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar);  
                else
                  IntegrateRayLinear(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar, offset,
                                     dev_debugdata3, dev_debugdata4, dev_debugdata5, dev_debugdata6,
                                     dev_debugdata7, dev_debugdata8, dev_debugdata9);
              }
              

              //Isosurface
              if(constmemory.isosurface > 0){
              //if(constmemory.isosurface && FindScalar(ray, intstep) >= cpscalar - EPSILON && FindScalar(ray, intstep) <= cpscalar + EPSILON){
                float c0 = ray->currentelem.interpolfunc1.w;
                float c1 = ray->currentelem.interpolfunc0.x;
                float c2 = ray->currentelem.interpolfunc0.y;
                float c3 = ray->currentelem.interpolfunc0.z;
                float c4 = ray->currentelem.interpolfunc0.w;
                float c5 = ray->currentelem.interpolfunc1.x;
                float c6 = ray->currentelem.interpolfunc1.y;
                float c7 = ray->currentelem.interpolfunc1.z;

                float4 pos = ray->eyepos + ray->dir * intstep;

                float3 N = normalize(make_float3(c1 + c4*pos.y + c5*pos.z + c7*pos.y*pos.z, c2 + c4*pos.x + c6*pos.z + c7*pos.x*pos.z, c3 + c5*pos.x + c6*pos.y + c7*pos.x*pos.y));

                float4 color = tex1D(texIsoColorScale, intstep);
                float3 L = normalize(make_float3(- ray->t * ray->dir));
                color.x *= abs(dot(N, L));
                color.y *= abs(dot(N, L));
                color.z *= abs(dot(N, L));


                color.x *= (color.w);
                color.y *= (color.w);
                color.z *= (color.w);
                ray->acccolor += (1.0f - ray->acccolor.w) * color;

                //dev_debugdata3[offset] = make_float4(ray->t, intstep, t, FindScalar(ray, intstep));
                //return;

              }
              
              ray->t = intstep;

              //break;
              //toutsideelement = false;
            }
          }
        }
        
        //return;
        //Integrate between last root and back face
        float diff = t - intstep;
        ray->frontscalar = FindScalar(ray, intstep);
        backscalar = FindScalar(ray, intstep + diff / constmemory.maxedge);
        if(constmemory.volumetric){
          if(constmemory.interpoltype == Const)
            IntegrateRayConst(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar);
          else
            IntegrateRayLinear(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar, offset,
                               dev_debugdata3, dev_debugdata4, dev_debugdata5, dev_debugdata6,
                               dev_debugdata7, dev_debugdata8, dev_debugdata9);
        }
        
        if(constmemory.debug2){
          if(countvalidroots == 1)
            ray->acccolor = make_float4(0, 0, 1, 1);
          else if(countvalidroots == 2)
            ray->acccolor = make_float4(0, 1, 0, 1);
          else if(countvalidroots == 3)
            ray->acccolor = make_float4(1, 0, 0, 1);
        }
        
        //return;
        //currentcp++;
        //cpscalar = tex1D(texVolumetricControlPoints, currentcp).x;
        currentcp++;
        //cpscalar = 0.3f;
        //if(currentcp >= 3) return;
      }
      */
      
#else


      float diffcpfront = 3.0f;
      float cpscalar = 3.0f;
      float volcpscalar = 3.0f;
      if(constmemory.volumetric)
      volcpscalar = FindControlPoint(ray, backscalar, tex1D(texVolumetricControlPoints, ray->frontscalar));

      float diffvolcpfront = fabs(volcpscalar - ray->frontscalar);

#ifdef CUDARC_ISOSURFACE
      float isocpscalar = 3.0f;
      if(constmemory.isosurface)
      isocpscalar = FindControlPoint(ray, backscalar, tex1D(texIsoControlPoints, ray->frontscalar));

      //Find which control point is the smaller one (cp on the isosurface or cp on the volumetric transfer function)
      float diffisocpfront = fabs(isocpscalar - ray->frontscalar);
      if(diffisocpfront < diffvolcpfront){
        diffcpfront = diffisocpfront;
        cpscalar = isocpscalar;
      }
      else{
        diffcpfront = diffvolcpfront;
        cpscalar = volcpscalar;
      }
#else
      diffcpfront = diffvolcpfront;
      cpscalar = volcpscalar;
#endif
      //Find if tetra contains cp
      float diffbackfront = fabs(backscalar - ray->frontscalar);

      

      if(diffcpfront < diffbackfront){
        t = FindIntegrationStep(ray, t, diffcpfront, diffbackfront, cpscalar, offset, dev_debugdata0, dev_debugdata1, dev_debugdata2);
        //Integrate between front and iso, but dont traverse to the next element
        backid = ray->frontid;
        backfaceid = ray->frontface;
        backscalar = cpscalar;
      }

      //dev_debugdata0[offset] = make_float4(backscalar, ray->frontscalar, cpscalar, isocpscalar);
      //dev_debugdata1[offset] = make_float4(diffcpfront, diffbackfront, 0, 0);

      
      //Probe box intersection
#ifdef CUDARC_PROBE_BOX
      if(constmemory.probebox > 0)
        IntersectProbeBox(ray, &t, &backid, &backscalar, probeboxmin, probeboxmax);
#endif
      
      

      //Volumetric
      if(constmemory.volumetric > 0){
        float diff = t - ray->t;
        if(constmemory.debug0 == 0 && diff > 0){
        //if(diff > 0){

          if(constmemory.interpoltype == Const){
            IntegrateRayConst(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar);
          }
          else{
            IntegrateRayLinear(ray, ray->eyepos, diff / constmemory.maxedge, backscalar, ray->frontscalar, offset,
                               dev_debugdata3, dev_debugdata4, dev_debugdata5, dev_debugdata6,
                               dev_debugdata7, dev_debugdata8, dev_debugdata9);
          }
        }
      }

      //Isosurface
#ifdef CUDARC_ISOSURFACE
      if(constmemory.isosurface > 0 && backid == ray->frontid && cpscalar == isocpscalar){

#ifdef CUDARC_GRADIENT_PERVERTEX
        //Initialize barycentric interpolation
        float4 gradv0   = tex1Dfetch(texGrad0, ray->frontid);
        float4 gradv1   = tex1Dfetch(texGrad1, ray->frontid);
        float4 gradv2   = tex1Dfetch(texGrad2, ray->frontid);
        float4 gradv3   = tex1Dfetch(texGrad3, ray->frontid);
        float4 gradient = (gradv0 + gradv1 + gradv2 + gradv3)/4.0;
        float3 N = normalize(make_float3(gradient.x,gradient.y,gradient.z));
#else
        float3 N = normalize(make_float3(ray->currentelem.interpolfunc0));
#endif

        float4 color = tex1D(texIsoColorScale, backscalar);
        float3 L = normalize(make_float3(- ray->t * ray->dir));
        color.x *= abs(dot(N, L));
        color.y *= abs(dot(N, L));
        color.z *= abs(dot(N, L));



#ifdef CUDARC_WHITE
        color.x = (1.0f - color.x) * (color.w);
        color.y = (1.0f - color.y) * (color.w);
        color.z = (1.0f - color.z) * (color.w);
        ray->acccolor += (1.0f - ray->acccolor.w) * make_float4(-color.x, -color.y, -color.z, color.w);
#else
        color.x *= (color.w);
        color.y *= (color.w);
        color.z *= (color.w);
        ray->acccolor += (1.0f - ray->acccolor.w) * color;
#endif
      }
      
#endif

      

#endif
    }


    //ray->acccolor.w = 1;

    //Traverse
    ray->frontid = backid;
    ray->frontface = backfaceid;
    ray->frontscalar = backscalar;
    ray->t = t;

    ray->currentelem.interpolfunc0 = tex1Dfetch(texInterpolFunc0, ray->frontid);
    ray->currentelem.adj0 = tex1Dfetch(texAdj0, ray->frontid);
#ifdef CUDARC_HEX
    ray->currentelem.interpolfunc1 = tex1Dfetch(texInterpolFunc1, ray->frontid);
    ray->currentelem.adj1 = tex1Dfetch(texAdj1, ray->frontid);
#endif
    t = CUDART_INF_F;
    backid = 0;
    backfaceid = 0;
    round = 0;
  }
}

/**
* Init CUDA variables
*/
void init(GLuint p_handleTexIntersect, GLuint p_handlePboOutput){




  //Prop
  cudaDeviceProp prop;
  int dev;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 1;
  CUDA_SAFE_CALL(cudaChooseDevice(&dev, &prop));
  CUDA_SAFE_CALL(cudaGLSetGLDevice( dev ));


  //Register output buffer
  CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cudaPboHandleOutput, p_handlePboOutput, cudaGraphicsMapFlagsNone));


  //Debug
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debugdata0, 800 * 800 * sizeof(float4)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debugdata1, 800 * 800 * sizeof(float4)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debugdata2, 800 * 800 * sizeof(float4)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debugdata3, 800 * 800 * sizeof(float4)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debugdata4, 800 * 800 * sizeof(float4)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debugdata5, 800 * 800 * sizeof(float4)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debugdata6, 800 * 800 * sizeof(float4)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debugdata7, 800 * 800 * sizeof(float4)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debugdata8, 800 * 800 * sizeof(float4)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debugdata9, 800 * 800 * sizeof(float4)));

}

/**
* CUDA callback (device)
*/
__global__ void Run(int depthPeelPass, float4 eyePos, float3 probeboxmin, float3 probeboxmax,
                    float4* dev_outputData, float4* dev_debugdata0, float4* dev_debugdata1,
                    float4* dev_debugdata2, float4* dev_debugdata3, float4* dev_debugdata4,
                    float4* dev_debugdata5, float4* dev_debugdata6, float4* dev_debugdata7,
                    float4* dev_debugdata8, float4* dev_debugdata9){

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int offset = x + y * blockDim.x * gridDim.x;

  dev_debugdata0[offset] = make_float4(0);
  dev_debugdata1[offset] = make_float4(0);
  dev_debugdata2[offset] = make_float4(0);
  dev_debugdata3[offset] = make_float4(0);
  dev_debugdata4[offset] = make_float4(0);
  dev_debugdata5[offset] = make_float4(0);
  dev_debugdata6[offset] = make_float4(0);
  dev_debugdata7[offset] = make_float4(0);
  dev_debugdata8[offset] = make_float4(0);
  dev_debugdata9[offset] = make_float4(0);

  Ray threadRay = Initialize(x, y, offset, eyePos);

  if(depthPeelPass > 0)
    threadRay.acccolor = dev_outputData[offset];
  

  if(threadRay.frontid > 0)
    Traverse(x, y, offset, &threadRay, probeboxmin, probeboxmax, dev_debugdata0, dev_debugdata1, dev_debugdata2, dev_debugdata3, dev_debugdata4, dev_debugdata5, dev_debugdata6, dev_debugdata7, dev_debugdata8, dev_debugdata9);


  dev_outputData[offset] = threadRay.acccolor;
  //dev_debugdata0[offset] = threadRay.acccolor;
}

/**
* CUDA callback (host)
*/
void run(float* p_kernelTime, float* p_overheadTime, int depthPeelPass, float* p_eyePos, float* probeboxmin, float* probeboxmax, int handleTexIntersect){

#ifdef CUDARC_TIME
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL(cudaEventCreate(&start));
  CUDA_SAFE_CALL(cudaEventCreate(&stop));
  CUDA_SAFE_CALL(cudaEventRecord(start, 0));
#endif

  //Register intersect buffer
  //TODO: do it every frame? Possible over-head?
  CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&cudaTexHandleIntersect, handleTexIntersect, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

  size_t size;
  CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cudaPboHandleOutput, NULL));
  CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&dev_outputData, &size, cudaPboHandleOutput));

  CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cudaTexHandleIntersect, NULL));
  CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&dev_intersectData, cudaTexHandleIntersect, 0, 0));


  //TODO: replace the rounding of the texIntersect values with the following code
  texIntersect.addressMode[0] = cudaAddressModeClamp;
  texIntersect.addressMode[1] = cudaAddressModeClamp;
  texIntersect.filterMode = cudaFilterModePoint;
  texIntersect.normalized = false;
  CUDA_SAFE_CALL(cudaBindTextureToArray(texIntersect, dev_intersectData));

  texVolumetricColorScale.addressMode[0] = cudaAddressModeClamp;
  texVolumetricColorScale.filterMode = cudaFilterModeLinear;
  texVolumetricColorScale.normalized = true;
  CUDA_SAFE_CALL(cudaBindTextureToArray(texVolumetricColorScale, dev_volcolorscale));

  texIsoColorScale.addressMode[0] = cudaAddressModeClamp;
  texIsoColorScale.filterMode = cudaFilterModeLinear;
  texIsoColorScale.normalized = true;
  CUDA_SAFE_CALL(cudaBindTextureToArray(texIsoColorScale, dev_isocolorscale));

  texVolumetricControlPoints.addressMode[0] = cudaAddressModeClamp;
#ifdef CUDARC_HEX
  texVolumetricControlPoints.filterMode = cudaFilterModePoint;
  texVolumetricControlPoints.normalized = false;
#else
  texVolumetricControlPoints.filterMode = cudaFilterModePoint;
  texVolumetricControlPoints.normalized = true;
#endif
  CUDA_SAFE_CALL(cudaBindTextureToArray(texVolumetricControlPoints, dev_volcontrolpoints));

  texIsoControlPoints.addressMode[0] = cudaAddressModeClamp;
  texIsoControlPoints.filterMode = cudaFilterModePoint;
  texIsoControlPoints.normalized = true;
  CUDA_SAFE_CALL(cudaBindTextureToArray(texIsoControlPoints, dev_isocontrolpoints));


  texZetaPsiGamma.addressMode[0] = cudaAddressModeClamp;
  texZetaPsiGamma.addressMode[1] = cudaAddressModeClamp;
  texZetaPsiGamma.filterMode = cudaFilterModeLinear;
  texZetaPsiGamma.normalized = true;
  CUDA_SAFE_CALL(cudaBindTextureToArray(texZetaPsiGamma, dev_zetaPsiGamma));

#ifdef CUDARC_TIME
  CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop));
  CUDA_SAFE_CALL(cudaEventElapsedTime(p_overheadTime, start, stop));

  CUDA_SAFE_CALL(cudaEventRecord(start, 0));
#endif

  //Kernel call
  Run<<<grids, threads>>>(depthPeelPass, 
                          make_float4(p_eyePos[0], p_eyePos[1], p_eyePos[2], 1),
                          make_float3(probeboxmin[0], probeboxmin[1], probeboxmin[2]),
                          make_float3(probeboxmax[0], probeboxmax[1], probeboxmax[2]),
                          dev_outputData, dev_debugdata0, dev_debugdata1, dev_debugdata2, dev_debugdata3, dev_debugdata4, dev_debugdata5, dev_debugdata6, dev_debugdata7, dev_debugdata8, dev_debugdata9);


  CUDA_SAFE_CALL(cudaGraphicsUnmapResources( 1, &cudaPboHandleOutput, NULL ) );
  CUDA_SAFE_CALL(cudaGraphicsUnmapResources( 1, &cudaTexHandleIntersect, NULL ) );
  CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cudaTexHandleIntersect));


#ifdef CUDARC_TIME
  CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop));
  CUDA_SAFE_CALL(cudaEventElapsedTime(p_kernelTime, start, stop));

  CUDA_SAFE_CALL(cudaEventDestroy(start));
  CUDA_SAFE_CALL(cudaEventDestroy(stop));
#endif

#ifdef CUDARC_VERBOSE
  cudaError_t cudaLastError = cudaGetLastError();
  if(cudaLastError != cudaSuccess)
    printf("%s\n", cudaGetErrorString(cudaLastError));
#endif

}

/**
* Create adj textures on the GPU
*/
void createGPUAdjTex(int index, int size, float* data){
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_adj[index], size));
  CUDA_SAFE_CALL(cudaMemcpy(dev_adj[index], data, size, cudaMemcpyHostToDevice));

  switch(index)
  {
  case 0:
    CUDA_SAFE_CALL(cudaBindTexture(0, texAdj0, dev_adj[index], size));
    break;
#ifdef CUDARC_HEX
  case 1:
    CUDA_SAFE_CALL(cudaBindTexture(0, texAdj1, dev_adj[index], size));
    break;
#endif
  default:
    break;
  }

#ifdef CUDARC_VERBOSE
  printf("Adj. to CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif
}


/**
* Create gradient vertex textures on the GPU
*/
#ifdef CUDARC_GRADIENT_PERVERTEX
void createGPUGradientVertexTex(int ni, int size, float* data)
{
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_gradientVertex[ni], size));
  CUDA_SAFE_CALL(cudaMemcpy(dev_gradientVertex[ni], data, size, cudaMemcpyHostToDevice));

  switch(ni)
  {
  case 0:
    CUDA_SAFE_CALL(cudaBindTexture(0, texGrad0, dev_gradientVertex[ni], size));
    break;
  case 1:
    CUDA_SAFE_CALL(cudaBindTexture(0, texGrad1, dev_gradientVertex[ni], size));
    break;
  case 2:
    CUDA_SAFE_CALL(cudaBindTexture(0, texGrad2, dev_gradientVertex[ni], size));
    break;
  case 3:
    CUDA_SAFE_CALL(cudaBindTexture(0, texGrad3, dev_gradientVertex[ni], size));
    break;
  default:
    break;
  }

#ifdef CUDARC_VERBOSE
  printf("Gradient vertex to CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif

}
#endif

/**
* Create node textures on the GPU
*/
#if defined(CUDARC_PLUCKER) || defined(CUDARC_BILINEAR)
void createGPUCollisionTex(int ni, int size, float* data){
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_collision[ni], size));
  CUDA_SAFE_CALL(cudaMemcpy(dev_collision[ni], data, size, cudaMemcpyHostToDevice));

  switch(ni)
  {
  case 0:
    CUDA_SAFE_CALL(cudaBindTexture(0, texNode0, dev_collision[ni], size));
    break;
  case 1:
    CUDA_SAFE_CALL(cudaBindTexture(0, texNode1, dev_collision[ni], size));
    break;
  case 2:
    CUDA_SAFE_CALL(cudaBindTexture(0, texNode2, dev_collision[ni], size));
    break;
  case 3:
    CUDA_SAFE_CALL(cudaBindTexture(0, texNode3, dev_collision[ni], size));
    break;
#ifdef CUDARC_HEX
  case 4:
    CUDA_SAFE_CALL(cudaBindTexture(0, texNode4, dev_collision[ni], size));
    break;
  case 5:
    CUDA_SAFE_CALL(cudaBindTexture(0, texNode5, dev_collision[ni], size));
    break;
  case 6:
    CUDA_SAFE_CALL(cudaBindTexture(0, texNode6, dev_collision[ni], size));
    break;
  case 7:
    CUDA_SAFE_CALL(cudaBindTexture(0, texNode7, dev_collision[ni], size));
    break;
#endif
  default:
    break;
  }

#ifdef CUDARC_VERBOSE
  printf("Collision data to CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif

}
#else
/**
* Create face textures on the GPU
*/
void createGPUCollisionTex(int fi, int size, float* data){
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_collision[fi], size));
  CUDA_SAFE_CALL(cudaMemcpy(dev_collision[fi], data, size, cudaMemcpyHostToDevice));

  switch(fi)
  {
  case 0:
    CUDA_SAFE_CALL(cudaBindTexture(0, texFace0Eq, dev_collision[fi], size));
    break;
  case 1:
    CUDA_SAFE_CALL(cudaBindTexture(0, texFace1Eq, dev_collision[fi], size));
    break;
  case 2:
    CUDA_SAFE_CALL(cudaBindTexture(0, texFace2Eq, dev_collision[fi], size));
    break;
  case 3:
    CUDA_SAFE_CALL(cudaBindTexture(0, texFace3Eq, dev_collision[fi], size));
    break;
#ifdef CUDARC_HEX
  case 4:
    CUDA_SAFE_CALL(cudaBindTexture(0, texFace4Eq, dev_collision[fi], size));
    break;
  case 5:
    CUDA_SAFE_CALL(cudaBindTexture(0, texFace5Eq, dev_collision[fi], size));
    break;
#endif
  default:
    break;
  }

#ifdef CUDARC_VERBOSE
  printf("Collision data to CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif

}
#endif

/**
* Create interpolation function textures on the GPU
*/
void createGPUInterpolFuncTex(int index, int size, float* data){
  CUDA_SAFE_CALL(cudaMalloc((void**)&dev_interpolfunc[index], size));
  CUDA_SAFE_CALL(cudaMemcpy(dev_interpolfunc[index], data, size, cudaMemcpyHostToDevice));

  switch(index)
  {
  case 0:
    CUDA_SAFE_CALL(cudaBindTexture(0, texInterpolFunc0, dev_interpolfunc[index], size));
    break;
#ifdef CUDARC_HEX
  case 1:
    CUDA_SAFE_CALL(cudaBindTexture(0, texInterpolFunc1, dev_interpolfunc[index], size));
    break;
#endif
  default:
    break;
  }

#ifdef CUDARC_VERBOSE
  printf("Interpolation func. to CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif
}

/**
* Create color scale texture on the GPU
*/
void createGPUColorScaleTex(int numValues, int size, float* volcolorscaledata, float* isocolorscale){
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

  CUDA_SAFE_CALL(cudaMallocArray(&dev_volcolorscale, &channelDesc, numValues));
  CUDA_SAFE_CALL(cudaMemcpyToArray(dev_volcolorscale, 0, 0, volcolorscaledata, size, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaBindTextureToArray(texVolumetricColorScale, dev_volcolorscale, channelDesc));  

  CUDA_SAFE_CALL(cudaMallocArray(&dev_isocolorscale, &channelDesc, numValues));
  CUDA_SAFE_CALL(cudaMemcpyToArray(dev_isocolorscale, 0, 0, isocolorscale, size, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaBindTextureToArray(texIsoColorScale, dev_isocolorscale, channelDesc));  

#ifdef CUDARC_VERBOSE
  printf("Color scales to CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif
}

/**
* Create control points texture on the GPU
*/
void createGPUVolControlPointsTex(int numValues, int size, float* data){
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
  CUDA_SAFE_CALL(cudaMallocArray(&dev_volcontrolpoints, &channelDesc, numValues));
  CUDA_SAFE_CALL(cudaMemcpyToArray(dev_volcontrolpoints, 0, 0, data, size, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaBindTextureToArray(texVolumetricColorScale, dev_volcontrolpoints, channelDesc)); 

#ifdef CUDARC_VERBOSE
  printf("Control points to CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif
}

void createGPUIsoControlPointsTex(int numValues, int size, float* data){
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
  CUDA_SAFE_CALL(cudaMallocArray(&dev_isocontrolpoints, &channelDesc, numValues));
  CUDA_SAFE_CALL(cudaMemcpyToArray(dev_isocontrolpoints, 0, 0, data, size, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaBindTextureToArray(texIsoColorScale, dev_isocontrolpoints, channelDesc)); 

#ifdef CUDARC_VERBOSE
  printf("Control points to CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif
}

/**
* Create zetapsigamma texture on the GPU
*/
void createGPUZetaPsiGammaTex(int numValues, int size, float* data){
#ifdef CUDARC_HEX
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
#else
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
#endif

  CUDA_SAFE_CALL(cudaMallocArray(&dev_zetaPsiGamma, &channelDesc, numValues, numValues));
  CUDA_SAFE_CALL(cudaMemcpyToArray(dev_zetaPsiGamma, 0, 0, data, size, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaBindTextureToArray(texZetaPsiGamma, dev_zetaPsiGamma, channelDesc)); 

#ifdef CUDARC_VERBOSE
  printf("PsiGamma to CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif
}

/**
* Delete textures from the GPU
*/
void deleteGPUTextures(int numAdjTex, int numInterpolFuncTex){

  for(int i=0; i<numAdjTex; i++){
    if(i==0){
      CUDA_SAFE_CALL(cudaUnbindTexture(texAdj0));
    }
#ifdef CUDARC_HEX
    else if(i==1){
      CUDA_SAFE_CALL(cudaUnbindTexture(texAdj1));
    }
#endif
    CUDA_SAFE_CALL(cudaFree(dev_collision[i]));
  }

  for(int i=0; i<numInterpolFuncTex; i++){

    if(i==0){
      CUDA_SAFE_CALL(cudaUnbindTexture(texInterpolFunc0));
    }
#ifdef CUDARC_HEX
    else if(i==1){
      CUDA_SAFE_CALL(cudaUnbindTexture(texInterpolFunc1));
    }
#endif
    CUDA_SAFE_CALL(cudaFree(dev_collision[i]));
  }


  //Color scale
  CUDA_SAFE_CALL(cudaUnbindTexture(texVolumetricColorScale));
  CUDA_SAFE_CALL(cudaFree(dev_volcolorscale));

  //Iso surfaces
  CUDA_SAFE_CALL(cudaUnbindTexture(texVolumetricControlPoints));
  CUDA_SAFE_CALL(cudaFree(dev_volcontrolpoints));

  //Psi Gamma table
  CUDA_SAFE_CALL(cudaUnbindTexture(texZetaPsiGamma));
  CUDA_SAFE_CALL(cudaFree(dev_zetaPsiGamma));


}

/**
* Print memory usage
*/
void printInfoGPUMemory(){
  size_t free, total;
  cuMemGetInfo(&free, &total);
  printf("#GPU Mem Info: Free = %d (%f), Total = %d\n", free, (float)free/(float)total, total);
}

/**
* Returns if the GPU supports CUDA
*/
bool isSupported(){
  int count;
  cudaGetDeviceCount(&count);

  if(count == 0)
    return false;

  //If count == 1, we have to check if device 0 is not just an emulation device
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, count);

  if(prop.major >= 1)
    return true;
  else
    return false;
}


/**
* Set const memory values
*/
void update(int p_blocksizex, int p_blocksizey, int p_winsizex, int p_winsizey,
                       bool p_debug0, bool p_debug1, bool p_debug2, float p_maxedge, int p_interpoltype, int p_numquadpoints, 
                       int p_numsteps, int p_nummaxcp, int p_numtraverses, int p_numelem,
                       bool p_isosurface, bool p_volumetric, bool p_probebox){

  grids = dim3(p_winsizex / p_blocksizex, p_winsizey / p_blocksizey);
  threads = dim3(p_blocksizex, p_blocksizey);

  ConstMemory *tempconstmemory = (ConstMemory*)malloc( sizeof(ConstMemory));
  tempconstmemory->numTets = p_numelem;
  tempconstmemory->screenSize = make_float2(p_winsizex, p_winsizey);
  tempconstmemory->maxedge = p_maxedge;
  tempconstmemory->interpoltype = p_interpoltype;
  tempconstmemory->numquadpoints = p_numquadpoints;
  tempconstmemory->numsteps = p_numsteps;
  tempconstmemory->numtraverses = p_numtraverses;
  tempconstmemory->nummaxcp = p_nummaxcp;
  tempconstmemory->debug0 = p_debug0;
  tempconstmemory->debug1 = p_debug1;
  tempconstmemory->debug2 = p_debug2;
  tempconstmemory->isosurface = p_isosurface;
  tempconstmemory->volumetric = p_volumetric;
  tempconstmemory->probebox = p_probebox;

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(constmemory, tempconstmemory, sizeof(ConstMemory)));
  delete tempconstmemory;
}

/**
* Debug
*/
void printDebugTexture(int x, int y){

  float* debug0 = new float[4 * 800 * 600 * sizeof(float)];
  float* debug1 = new float[4 * 800 * 600 * sizeof(float)];
  float* debug2 = new float[4 * 800 * 600 * sizeof(float)];
  float* debug3 = new float[4 * 800 * 600 * sizeof(float)];
  float* debug4 = new float[4 * 800 * 600 * sizeof(float)];
  float* debug5 = new float[4 * 800 * 600 * sizeof(float)];
  float* debug6 = new float[4 * 800 * 600 * sizeof(float)];
  float* debug7 = new float[4 * 800 * 600 * sizeof(float)];
  float* debug8 = new float[4 * 800 * 600 * sizeof(float)];
  float* debug9 = new float[4 * 800 * 600 * sizeof(float)];

  CUDA_SAFE_CALL(cudaMemcpy2D(debug0, 800 * sizeof(float4), dev_debugdata0, 800 * sizeof(float4), 800 * sizeof(float4), 600, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(debug1, 800 * sizeof(float4), dev_debugdata1, 800 * sizeof(float4), 800 * sizeof(float4), 600, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(debug2, 800 * sizeof(float4), dev_debugdata2, 800 * sizeof(float4), 800 * sizeof(float4), 600, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(debug3, 800 * sizeof(float4), dev_debugdata3, 800 * sizeof(float4), 800 * sizeof(float4), 600, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(debug4, 800 * sizeof(float4), dev_debugdata4, 800 * sizeof(float4), 800 * sizeof(float4), 600, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(debug5, 800 * sizeof(float4), dev_debugdata5, 800 * sizeof(float4), 800 * sizeof(float4), 600, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(debug6, 800 * sizeof(float4), dev_debugdata6, 800 * sizeof(float4), 800 * sizeof(float4), 600, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(debug7, 800 * sizeof(float4), dev_debugdata7, 800 * sizeof(float4), 800 * sizeof(float4), 600, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(debug8, 800 * sizeof(float4), dev_debugdata8, 800 * sizeof(float4), 800 * sizeof(float4), 600, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(debug9, 800 * sizeof(float4), dev_debugdata9, 800 * sizeof(float4), 800 * sizeof(float4), 600, cudaMemcpyDeviceToHost));

  printf("\n(x: %d, y: %d)\n", x, y);
  printf("dir: %.10f, %.10f, %.10f, %.10f\n", debug0[4 * (800 * y + x)], debug0[4 * (800 * y + x) + 1], debug0[4 * (800 * y + x) + 2], debug0[4 * (800 * y + x) + 3]);
  printf("tri. eq.: %.10f, %.10f, %.10f, %.10f\n", debug1[4 * (800 * y + x)], debug1[4 * (800 * y + x) + 1], debug1[4 * (800 * y + x) + 2], debug1[4 * (800 * y + x) + 3]);
  printf("roots: %.10f, %.10f, %.10f, n. roots: %.10f\n", debug2[4 * (800 * y + x)], debug2[4 * (800 * y + x) + 1], debug2[4 * (800 * y + x) + 2], debug2[4 * (800 * y + x) + 3]);
  printf("ray->t: %.10f, t: %.10f, D: %.10f, %.10f\n", debug3[4 * (800 * y + x)], debug3[4 * (800 * y + x) + 1], debug3[4 * (800 * y + x) + 2], debug3[4 * (800 * y + x) + 3]);
  printf("w: %.10f, %.10f, %.10f, %.10f\n", debug4[4 * (800 * y + x)], debug4[4 * (800 * y + x) + 1], debug4[4 * (800 * y + x) + 2], debug4[4 * (800 * y + x) + 3]);
  printf("zeta: %.10f, psi: %.10f, division: %.10f, %.10f\n", debug5[4 * (800 * y + x)], debug5[4 * (800 * y + x) + 1], debug5[4 * (800 * y + x) + 2], debug5[4 * (800 * y + x) + 3]);
  printf("sf_p: %.10f, sb_p: %.10f, sf_t: %.10f, sb_t: %.10f\n", debug6[4 * (800 * y + x)], debug6[4 * (800 * y + x) + 1], debug6[4 * (800 * y + x) + 2], debug6[4 * (800 * y + x) + 3]);
  printf("eye: %.10f, %.10f, %.10f, %.10f\n", debug7[4 * (800 * y + x)], debug7[4 * (800 * y + x) + 1], debug7[4 * (800 * y + x) + 2], debug7[4 * (800 * y + x) + 3]);
  printf("c0: %.10f, c1: %.10f, c2: %.10f, c3: %.10f\n", debug8[4 * (800 * y + x)], debug8[4 * (800 * y + x) + 1], debug8[4 * (800 * y + x) + 2], debug8[4 * (800 * y + x) + 3]);
  printf("c4: %.10f, c5: %.10f, c6: %.10f, c7: %.10f\n", debug9[4 * (800 * y + x)], debug9[4 * (800 * y + x) + 1], debug9[4 * (800 * y + x) + 2], debug9[4 * (800 * y + x) + 3]);

  delete [] debug0;
  delete [] debug1;
  delete [] debug2;
  delete [] debug3;
  delete [] debug4;
  delete [] debug5;
  delete [] debug6;
  delete [] debug7;
  delete [] debug8;
  delete [] debug9;
}
