/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* cudarc.cu: CUDA functions
*/

#define EPSILON 0

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
#include "cudamemory.cuh"

enum InterpolType {Const = 0, Linear = 1, Quad = 2, Step = 3};

struct Elem{
  float4 interpolfunc0;
#ifdef CUDARC_HEX
  float4 interpolfunc1;
#endif
};

struct Ray{
  float4 acccolor;
  float4 dir;
  float4 eyepos;
  float t;
  int frontid;
  int frontface;
  float frontscalar;
  Elem currentelem;
};

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __host__ __device__ float4 cross(float4 a, float4 b)
{ 
  return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0); 
}


inline __host__ __device__ float permuted_inner_produtct(float4 pr, float4 qr, float4 ps, float4 qs)
{
  return dot(pr, qs) + dot(qr, ps);
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

__device__ float2 Intersect(Ray* ray, float4 v00, float4 v01, float4 v10, float4 v11, bool getfart){


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
* Calculate ZetaPsi, using gaussian quadrature (hexahedral mesh)
*/
__device__ float4 GetZetaPsiQuad(Ray* ray, float4 eyepos, float raySegLength, float alphaBack, float alphaFront, float sback, float sfront){

  float pf = alphaBack;
  float pb = alphaFront;
  float D = raySegLength;
  
  float a1 = eyepos.x;
  float a2 = ray->dir.x;
  float b1 = eyepos.y;
  float b2 = ray->dir.y;
  float g1 = eyepos.z;
  float g2 = ray->dir.z;

  float c0 = ray->currentelem.interpolfunc1.w;
  float c1 = ray->currentelem.interpolfunc0.x;
  float c2 = ray->currentelem.interpolfunc0.y;
  float c3 = ray->currentelem.interpolfunc0.z;
  float c4 = ray->currentelem.interpolfunc0.w;
  float c5 = ray->currentelem.interpolfunc1.x;
  float c6 = ray->currentelem.interpolfunc1.y;
  float c7 = ray->currentelem.interpolfunc1.z;

  float4 w; //{w0, w1, w2, w3}
  w.x = c0 + a1*c1 + b1*c2 + a1*b1*c4 + c3*g1 + a1*c5*g1 + b1*c6*g1 + a1*b1*c7*g1;
  w.y = a2*c1 + b2*c2 + a2*b1*c4 + a1*b2*c4 + a2*c5*g1 + b2*c6*g1 + a2*b1*c7*g1 + a1*b2*c7*g1 + c3*g2 + a1*c5*g2 + b1*c6*g2 + a1*b1*c7*g2;
  w.z = a2*b2*c4 + a2*b2*c7*g1 + a2*c5*g2 + b2*c6*g2 + a2*b1*c7*g2 + a1*b2*c7*g2;
  w.w = a2*b2*c7*g2;
  /*
  w.x = fabs(w.x);
  w.y = fabs(w.y);
  w.z = fabs(w.z);
  w.w = fabs(w.w);
  */

  //w += make_float4(1.0f);

  float zeta = expf(- (6.0f * D * (pb + pf) * w.y + 4.0f * D * D * (pb + 2.0f * pf) * w.z + 3.0f * D * D * D * (pb + 3.0f * pf) * w.w) / (12.0f * w.y + 12.0f * D * w.z + 12.0f * D * D * w.w));

  float aux = (w.y * D + w.z * D * D + w.w * D * D * D);
  //Psi (using gauss quadrature)
  float4 psi1, psi2, exp1, exp2;
  float Doveraux = D / aux;
  
  exp1.x = - (Doveraux * (0.980145 * aux * pf + D * (pb - pf) * (0.499803 * w.y + D * (0.333331 * w.z + 0.25 * D * w.w))));
  exp1.y = - (Doveraux * (0.898333 * aux * pf + D * (pb - pf) * (0.494832 * w.y + D * (0.332983 * w.z + 0.249973 * D * w.w))));
  exp1.z = - (Doveraux * (0.762766 * aux * pf + D * (pb - pf) * (0.47186 * w.y + D * (0.328883 * w.z + 0.249208 * D * w.w))));
  exp1.w = - (Doveraux * (0.591717 * aux * pf + D * (pb - pf) * (0.416653 * w.y + D * (0.310647 * w.z + 0.243053 * D * w.w))));

  exp2.x = - (Doveraux * (0.408283 * aux * pf + D * (pb - pf) * (0.324935 * w.y + D * (0.264274 * w.z + 0.219352 * D * w.w))));
  exp2.y = - (Doveraux * (0.237234 * aux * pf + D * (pb - pf) * (0.209094 * w.y + D * (0.185404 * w.z + 0.165374 * D * w.w))));
  exp2.z = - (Doveraux * (0.101667 * aux * pf + D * (pb - pf) * (0.0964987 * w.y + D * (0.0916809 * w.z + 0.0871867 * D * w.w))));
  exp2.w = - (Doveraux * (0.0198551 * aux * pf + D * (pb - pf) * (0.019658 * w.y + D * (0.0194635 * w.z + 0.0192715 * D * w.w))));

  psi1.x = 0.0506143 * D * expf(exp1.x) * (w.y + 0.0397101 * D * w.z + 0.00118267 * D * D * w.w);
  psi1.y = 0.111191 * D * expf(exp1.y) * (w.y + 0.203334 * D * w.z + 0.0310084 * D * D * w.w);
  psi1.z = 0.156853 * D * expf(exp1.z) * (w.y + 0.474468 * D * w.z + 0.16884 * D * D * w.w);
  psi1.w = 0.181342 * D * expf(exp1.w) * (w.y + 0.816565 * D * w.z + 0.500084 * D * D * w.w);

  psi2.x = 0.181342 * D * expf(exp2.x) * (w.y + 1.18343 * D * w.z + 1.05039 * D * D * w.w);
  psi2.y = 0.156853 * D * expf(exp2.y) * (w.y + 1.52553 * D * w.z + 1.74544 * D * D * w.w);
  psi2.z = 0.111191 * D * expf(exp2.z) * (w.y + 1.79667 * D * w.z + 2.42101  * D * D * w.w);
  psi2.w = 0.0506143 * D * expf(exp2.w) * (w.y + 1.96029 * D * w.z + 2.88205 * D * D * w.w);
  
  /*
  psi1.x = 0.0506143 * expf(- Doveraux * (0.980145 * aux * pf + 0.084 * (pb - pf) * (5.99763 * D * w.y + 3.99997 * D * D * w.z - 3.0f * D * D * D * w.w))) * (w.y + 0.0397101 * w.z + 0.00118267 * w.w);
  psi1.y = 0.111191 * expf(- Doveraux * (0.898333 * aux * pf + 0.084 * (pb - pf) * (5.93798 * D * w.y + 3.9958 * D * D * w.z - 2.99968 * D * D * D * w.w))) * (w.y + 0.203334 * w.z + 0.0310084 * w.w);
  psi1.z = 0.156853 * expf(- Doveraux * (0.762766  * aux * pf + 0.084 * (pb - pf) * (5.66232 * D * w.y + 3.94659 * D * D * w.z - 2.9905 * D * D * D * w.w))) * (w.y + 0.474468 * w.z + 0.16884 * w.w);
  psi1.w = 0.181342 * expf(- Doveraux * (0.591717 * aux * pf + 0.084 * (pb - pf) * (4.99983 * D * w.y + 3.72777 * D * D * w.z - 2.91664 * D * D * D * w.w))) * (w.y + 0.816565 * w.z + 0.500084 * w.w);

  psi2.x = 0.181342 * expf(- Doveraux * (0.408283 * aux * pf + 0.084 * (pb - pf) * (3.89922  * D * w.y + 3.17129 * D * D * w.z - 2.63223 * D * D * D * w.w))) * (w.y + 1.18343 * w.z + 1.05039 * w.w);
  psi2.y = 0.156853 * expf(- Doveraux * (0.237234  * aux * pf + 0.084 * (pb - pf) * (2.50913  * D * w.y + 2.22485 * D * D * w.z - 1.98448 * D * D * D * w.w))) * (w.y + 1.52553 * w.z + 1.74544 * w.w);
  psi2.z = 0.111191 * expf(- Doveraux * (0.101667 * aux * pf + 0.084 * (pb - pf) * (1.15798  * D * w.y + 1.10017 * D * D * w.z - 1.04624 * D * D * D * w.w))) * (w.y + 1.79667 * w.z + 2.42101  * w.w);
  psi2.w = 0.0506143 * expf(- Doveraux * (0.0198551 * aux * pf + 0.084 * (pb - pf) * (0.235896 * D * w.y + 0.233561 * D * D * w.z - 0.231258 * D * D * D * w.w))) * (w.y + 1.96029 * w.z + 2.88205 * w.w);
  */
  float psi = (psi1.x + psi1.y + psi1.z + psi1.w + psi2.x + psi2.y + psi2.z + psi2.w);

  return make_float4(zeta, psi , psi, psi);
}

/**
* Calculate ZetaPsi, fetching psi gamma from texture (hexahedral mesh)
*/
__device__ float4 GetZetaPsiFetch(Ray* ray, float4 eyepos, float raySegLength, float alphaBack, float alphaFront, float sback, float sfront){
  /*
  float a1 = (eyepos.x+(ray->t)*ray->dir.x);
  float a2 = ray->dir.x;
  float b1 = (eyepos.y+(ray->t)*ray->dir.y);
  float b2 = ray->dir.y;
  float g1 = (eyepos.z+(ray->t)*ray->dir.z);
  float g2 = ray->dir.z;

  float c1 = ray->currentelem.interpolfunc1.w;
  float c2 = ray->currentelem.interpolfunc0.x;
  float c3 = ray->currentelem.interpolfunc0.y;
  float c4 = ray->currentelem.interpolfunc0.z;
  float c5 = ray->currentelem.interpolfunc0.w;
  float c6 = ray->currentelem.interpolfunc1.x;
  float c7 = ray->currentelem.interpolfunc1.y;
  float c8 = ray->currentelem.interpolfunc1.z;


  float4 w;
  w.x = c1 + a1*c2 + b1*c3 + a1*b1*c5 + c4*g1 + a1*c6*g1 + b1*c7*g1 + a1*b1*c8*g1;
  w.y = a2*c2 + b2*c3 + a2*b1*c5 + a1*b2*c5 + a2*c6*g1 + b2*c7*g1 + a2*b1*c8*g1 + a1*b2*c8*g1 + c4*g2 + a1*c6*g2 + b1*c7*g2 + a1*b1*c8*g2;
  w.z = a2*b2*c5 + a2*b2*c8*g1 + a2*c6*g2 + a2*c7*g2 + a2*b1*c8*g2 + a1*b2*c8*g2;
  w.w = a2*b2*c8*g2;

  float polyn = w.y + raySegLength * w.z + raySegLength * raySegLength * w.w;
  float3 num = (alphaBack - alphaFront) * make_float3(w.y, w.z, w.w) / polyn;

  float3 gamma = num / (num + 1.0f);
  float4 zetapsigamma = tex3D(texZetaPsiGamma, gamma.x, gamma.y, gamma.z) * expf(- raySegLength * alphaFront);
  zetapsigamma.y *= raySegLength;
  zetapsigamma.z *= raySegLength;
  zetapsigamma.w *= raySegLength;

  float psi = ((((w.y * zetapsigamma.y)) + ((2 * w.z * zetapsigamma.z)) + ((3 * w.w * zetapsigamma.w))) / polyn);

  return make_float4(zetapsigamma.x, psi, psi, psi);
  */
  return make_float4(0, 0, 0, 0);
}

/**
* Find scalar of the (x,y,z) point (hexahedral mesh)
*/
__device__ float FindScalar(Ray* ray, float p_t){

  float4 pos = ray->eyepos + p_t * ray->dir;
  pos.w = 1.0;

  float4 interpolfunc0 = ray->currentelem.interpolfunc0;
  float4 interpolfunc1 = ray->currentelem.interpolfunc1;

  return interpolfunc0.x * pos.x + interpolfunc0.y * pos.y + interpolfunc0.z * pos.z + interpolfunc0.w * pos.x * pos.y
    + interpolfunc1.x * pos.x * pos.z + interpolfunc1.y * pos.y * pos.z + interpolfunc1.z * pos.x * pos.y * pos.z + interpolfunc1.w;


}

/**
* Find integration step (hex)
*/
/*
inline __device__ float FindIntegrationStep(Ray* ray, float t, float diffcpfront, float diffbackfront){

  float c0 = ray->currentelem.interpolfunc1.w;
  float c1 = ray->currentelem.interpolfunc0.x;
  float c2 = ray->currentelem.interpolfunc0.y;
  float c3 = ray->currentelem.interpolfunc0.z;
  float c4 = ray->currentelem.interpolfunc0.w;
  float c5 = ray->currentelem.interpolfunc1.x;
  float c6 = ray->currentelem.interpolfunc1.y;
  float c7 = ray->currentelem.interpolfunc1.z;
  
  float ox = eyepos.x;
  float oy = eyepos.y;
  float oz = eyepos.z;

  float dx = threadray->dir.x;
  float dy = threadray->dir.y;
  float dz = threadray->dir.z;

  float a = c0 + c1*ox + c2*oy + c4*ox*oy + c3*oz + c6*ox*oz + c5*oy*oz + c7*ox*oy*oz;
  float b = c1*dx + c2*dy + c3*dz + c4*dy*ox + c6*dz*ox + c4*dx*oy + c5*dz*oy + c7*dz*ox*oy + c6*dx*oz + c5*dy*oz + c7*dy*ox*oz + c7*dx*oy*oz;
  float c = c4*dx*dy + c6*dx*dz + c5*dy*dz + c7*dy*dz*ox + c7*dx*dz*oy + c7*dx*dy*oz;
  float d = c7*dx*dy*dz - isoScalar;


  float delta = 18 * a * b * c * d + 4 * b * b * b * d + b * b + c * c - 4 * a * c * c * c - 27 * a * a * d * d;
  
}
*/

#else
/**
* Calculate ZetaPsi, using gaussian quadrature (tetrahedral mesh)
*/
__device__ float4 GetZetaPsiQuad(Ray* ray, float4 eyepos, float raySegLength, float alphaBack, float alphaFront, float sback, float sfront){
  float4 t1, weights1, expf_psi1, t2, weights2, expf_psi2;
  float psi;
  float2 alphaL; // alpha * rayLength
  alphaL = raySegLength * make_float2(alphaBack, alphaFront);

  //Zeta
  float zeta = expf(-dot(alphaL, make_float2(0.5f, 0.5f)));

  //Psi
  t1.x = 0.019855071751231912;
  t1.y = 0.10166676129318664;
  t1.z = 0.2372337950418355;
  t1.w = 0.40828267875217505;
  t2.x = 0.591717321247825;
  t2.y = 0.7627662049581645;
  t2.z = 0.8983332387068134;
  t2.w = 0.9801449282487681;
  weights1.x = 0.05061426814518863;
  weights1.y = 0.11119051722668714;
  weights1.z = 0.15685332293894347;
  weights1.w = 0.18134189168918077;
  weights2.x = 0.18134189168918077;
  weights2.y = 0.15685332293894347;
  weights2.z = 0.11119051722668714;
  weights2.w = 0.05061426814518863;

  expf_psi1 = - make_float4(raySegLength) * ((t1) * (alphaBack * (make_float4(1)-t1) + alphaFront * t1));
  expf_psi2 = - make_float4(raySegLength) * ((t2) * (alphaBack * (make_float4(1)-t2) + alphaFront * t2));

  psi = dot(make_float4(expf(expf_psi1.x), expf(expf_psi1.y), expf(expf_psi1.z), expf(expf_psi1.w)), weights1);  
  psi += dot(make_float4(expf(expf_psi2.x), expf(expf_psi2.y), expf(expf_psi2.z), expf(expf_psi2.w)), weights2);

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
inline __device__ float FindIntegrationStep(Ray* ray, float t, float diffcpfront, float diffbackfront){
  return (ray->t) + (t - ray->t) * (diffcpfront / diffbackfront);
  //t = (isoScalar - dot(threadRay->eyepos, threadRay->currentelem.interpolfunc0) - threadRay->currentelem.interpolfunc0.w) / (dot(threadRay->dir, threadRay->currentelem.interpolfunc0));
}

#endif

/**
* Find control point
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
}


/**
* Linear integration/trilinear of the ray
*/
__device__ void IntegrateRayLinear(Ray* ray, float4 eyepos, float raySegLength, float sback, float sfront){

  float3 color;
  float alpha;

  float4 colorBack = tex1D(texVolumetricColorScale, sback);
  float4 colorFront = tex1D(texVolumetricColorScale, sfront);
  float4 zetapsi;

  if(constMemory.interpoltype == Quad)
    zetapsi = GetZetaPsiQuad(ray, eyepos, raySegLength, colorBack.w, colorFront.w, sback, sfront);
  else
    zetapsi = GetZetaPsiFetch(ray, eyepos, raySegLength, colorBack.w, colorFront.w, sback, sfront);

  alpha = 1 - zetapsi.x;
  //Finally
#ifdef CUDARC_HEX
#ifdef CUDARC_WHITE
  color = (make_float3(1)- make_float3(colorFront)) * (zetapsi.y - zetapsi.x) + (make_float3(1) - make_float3(colorBack)) * (1.0f - zetapsi.y);
  ray->acccolor += (1 - ray->acccolor.w) * make_float4(-color.x, -color.y, -color.z, alpha);
#else
  color = make_float3(colorFront) * (zetapsi.y - zetapsi.x) + make_float3(colorBack) * (1.0f - zetapsi.y);
  ray->acccolor += (1 - ray->acccolor.w) * make_float4(color.x, color.y, color.z, alpha);
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
  /*
  if(zetapsi.y < 0)
    ray->acccolor = make_float4(1, 0, 0, 1);
  else if(zetapsi.y > 1.0f)
    ray->acccolor = make_float4(0, 1, 0, 1);
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
* Initialize function, calculate the starting position of the ray on the mesh
*/
__device__ Ray Initialize(int x, int y, int offset, float4 eyePos){

  float4 tetraInfo = tex2D(texIntersect, x, y);
  float4 dir = make_float4(tetraInfo.x, tetraInfo.y, tetraInfo.z, 0);
  int tid = floor(tetraInfo.w + 0.5f);

  Ray threadRay;
  threadRay.t = 0.0f;
  threadRay.dir = normalize(dir);
  threadRay.eyepos = eyePos;
#ifdef CUDARC_HEX
  threadRay.frontid = tid / 6; 
  threadRay.frontface = tid % 6; 
#else
  threadRay.frontid = tid / 4;  
  threadRay.frontface = tid % 4;  
#endif
#ifdef CUDARC_WHITE
  threadRay.acccolor = make_float4(1, 1, 1, 0);
#else
  threadRay.acccolor = make_float4(0);
#endif

  threadRay.currentelem.interpolfunc0 = tex1Dfetch(texInterpolFunc0, threadRay.frontid);
#ifdef CUDARC_HEX
  threadRay.currentelem.interpolfunc1 = tex1Dfetch(texInterpolFunc1, threadRay.frontid);
#endif


#ifdef CUDARC_BILINEAR
  float4 v0 = tex1Dfetch(texNode0, threadRay.frontid);
  float4 v1 = tex1Dfetch(texNode1, threadRay.frontid);
  float4 v2 = tex1Dfetch(texNode2, threadRay.frontid);
  float4 v3 = tex1Dfetch(texNode3, threadRay.frontid);
  float4 v4 = tex1Dfetch(texNode4, threadRay.frontid);
  float4 v5 = tex1Dfetch(texNode5, threadRay.frontid);
  float4 v6 = tex1Dfetch(texNode6, threadRay.frontid);
  float4 v7 = tex1Dfetch(texNode7, threadRay.frontid);

  if(threadRay.frontface == 0)
    threadRay.t = Intersect(&threadRay, v5, v4, v6, v7, false).x; //fem
    //threadRay.t = Intersect(&threadRay, v0, v1, v2, v3, false).x; //res
  else if(threadRay.frontface == 1)
    threadRay.t = Intersect(&threadRay, v0, v1, v3, v2, false).x; //fem
    //threadRay.t = Intersect(&threadRay, v4, v5, v6, v7, false).x; //res
  else if(threadRay.frontface == 2)
    threadRay.t = Intersect(&threadRay, v1, v2, v5, v6, false).x; //fem
    //threadRay.t = Intersect(&threadRay, v1, v3, v5, v7, false).x; //res
  else if(threadRay.frontface == 3)'
    threadRay.t = Intersect(&threadRay, v0, v3, v4, v7, false).x; //fem
    //threadRay.t = Intersect(&threadRay, v0, v2, v4, v6, false).x; //res
  else if(threadRay.frontface == 4)
    threadRay.t = Intersect(&threadRay, v0, v1, v4, v5, false).x; //fem
    //threadRay.t = Intersect(&threadRay, v0, v1, v4, v5, false).x; //res
  else if(threadRay.frontface == 5)
    threadRay.t = Intersect(&threadRay, v2, v3, v6, v7, false).x; //fem
    //threadRay.t = Intersect(&threadRay, v2, v3, v6, v7, false).x; //res

  if(threadRay.t == CUDART_INF_F)
    threadRay.acccolor = make_float4(0,0,0,1);

  //threadRay.t = length(dir);

#else
   threadRay.t = length(dir);
#endif

  threadRay.frontscalar = FindScalar(&threadRay, threadRay.t);
  /*
  if(threadRay.t == CUDART_INF_F)
    threadRay.acccolor = make_float4(1, 0, 0, 1);
  else
    threadRay.acccolor = make_float4(0, 0, 1, 1);
  */

  return threadRay;

}

/**
* Volumetric traverse the ray through the mesh
*/
__device__ void Traverse(int x, int y, int offset, Ray* threadRay){

  float4 planeEq;
  float sameDirection;
  float t = CUDART_INF_F;
  int backid = 0;
  int backfaceid = 0;
  float round = 0;

#ifdef CUDARC_HEX
  float4 hexAdj1 = tex1Dfetch(texAdj0, threadRay->frontid);
  float4 hexAdj2 = tex1Dfetch(texAdj1, threadRay->frontid);
#else
  float4 tetAdj = tex1Dfetch(texAdj0, threadRay->frontid);
#endif

#ifdef CUDARC_HEX
  float4 v0, v1, v2, v3, v4, v5, v6, v7;
  float4 ray = cross(threadRay->dir, threadRay->eyepos);
  /*
  float4 v0 = tex1Dfetch(texNode0, threadRay->frontid);
  float4 v1 = tex1Dfetch(texNode1, threadRay->frontid);
  float4 v2 = tex1Dfetch(texNode2, threadRay->frontid);
  float4 v3 = tex1Dfetch(texNode3, threadRay->frontid);
  float4 v4 = tex1Dfetch(texNode4, threadRay->frontid);
  float4 v5 = tex1Dfetch(texNode5, threadRay->frontid);
  float4 v6 = tex1Dfetch(texNode6, threadRay->frontid);
  float4 v7 = tex1Dfetch(texNode7, threadRay->frontid);
  float4 ray = cross(threadRay->dir, threadRay->eyepos);
  float4 point;
  
  float4 v02 = (v0 - v2);
  float4 q02 = cross(v02, v0);
  float ps02 = permuted_inner_produtct(v02, q02, threadRay->dir, ray);

  float4 v32 = (v3 - v2);
  float4 q32 = cross(v32, v2);
  float ps32 = permuted_inner_produtct(v32, q32, threadRay->dir, ray);

  float4 v13 = (v1 - v3);
  float4 q13 = cross(v13, v1);
  float ps13 = permuted_inner_produtct(v13, q13, threadRay->dir, ray);

  float4 v01 = (v0 - v1);
  float4 q01 = cross(v01, v0);
  float ps01 = permuted_inner_produtct(v01, q01, threadRay->dir, ray);

  float4 v21 = (v2 - v1);
  float4 q21 = cross(v21, v1);
  float ps21 = permuted_inner_produtct(v21, q21, threadRay->dir, ray);

  float4 v76 = (v7 - v6);
  float4 q76 = cross(v76, v6);
  float ps76 = permuted_inner_produtct(v76, q76, threadRay->dir, ray);

  float4 v57 = (v5 - v7);
  float4 q57 = cross(v57, v7);
  float ps57 = permuted_inner_produtct(v57, q57, threadRay->dir, ray);

  float4 v46 = (v4 - v6);
  float4 q46 = cross(v46, v6);
  float ps46 = permuted_inner_produtct(v46, q46, threadRay->dir, ray);

  float4 v45 = (v4 - v5);
  float4 q45 = cross(v45, v5);
  float ps45 = permuted_inner_produtct(v45, q45, threadRay->dir, ray);

  float4 v26 = (v2 - v6);
  float4 q26 = cross(v26, v6);
  float ps26 = permuted_inner_produtct(v26, q26, threadRay->dir, ray);

  float4 v37 = (v3 - v7);
  float4 q37 = cross(v37, v7);
  float ps37 = permuted_inner_produtct(v37, q37, threadRay->dir, ray);

  float4 v04 = (v0 - v4);
  float4 q04 = cross(v04, v4);
  float ps04 = permuted_inner_produtct(v04, q04, threadRay->dir, ray);

  float4 v15 = (v1 - v5);
  float4 q15 = cross(v15, v5);
  float ps15 = permuted_inner_produtct(v15, q15, threadRay->dir, ray);
  */
#else
#ifdef CUDARC_PLUCKER
  float4 v0 = tex1Dfetch(texNode0, threadRay->frontid);
  float4 v1 = tex1Dfetch(texNode1, threadRay->frontid);
  float4 v2 = tex1Dfetch(texNode2, threadRay->frontid);
  float4 v3 = tex1Dfetch(texNode3, threadRay->frontid);
  float4 ray = cross(threadRay->dir, threadRay->eyepos);
  float4 point;

  float4 v02 = (v0 - v2);
  float4 q02 = cross(v02, v0);
  float ps02 = permuted_inner_produtct(v02, q02, threadRay->dir, ray);

  float4 v32 = (v3 - v2);
  float4 q32 = cross(v32, v2);
  float ps32 = permuted_inner_produtct(v32, q32, threadRay->dir, ray);

  float4 v03 = (v0 - v3);
  float4 q03 = cross(v03, v0);
  float ps03 = permuted_inner_produtct(v03, q03, threadRay->dir, ray);

  float4 v13 = (v1 - v3);
  float4 q13 = cross(v13, v1);
  float ps13 = permuted_inner_produtct(v13, q13, threadRay->dir, ray);

  float4 v01 = (v0 - v1);
  float4 q01 = cross(v01, v0);
  float ps01 = permuted_inner_produtct(v01, q01, threadRay->dir, ray);

  float4 v21 = (v2 - v1);
  float4 q21 = cross(v21, v1);
  float ps21 = permuted_inner_produtct(v21, q21, threadRay->dir, ray);
#endif
#endif


  int aux = 0;
  while((constMemory.numtraverses > 0 && constMemory.debug) || (threadRay->frontid > 0 && threadRay->acccolor.w < 0.99)){
    
    if((constMemory.numtraverses > 0 && aux >= constMemory.numtraverses) || aux >= CUDARC_MAX_ITERATIONS)
      break;

    aux++;

    //t = -(P0 . N + d) / (V . N) (http://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm)
    //t = -(eyePos . normal + d) / (eyeDir . normal)
    threadRay->dir.w = 0;
    threadRay->eyepos.w = 1;

#ifdef CUDARC_HEX
#ifdef CUDARC_BILINEAR
    v0 = tex1Dfetch(texNode0, threadRay->frontid);
    v1 = tex1Dfetch(texNode1, threadRay->frontid);
    v2 = tex1Dfetch(texNode2, threadRay->frontid);
    v3 = tex1Dfetch(texNode3, threadRay->frontid);
    v4 = tex1Dfetch(texNode4, threadRay->frontid);
    v5 = tex1Dfetch(texNode5, threadRay->frontid);
    v6 = tex1Dfetch(texNode6, threadRay->frontid);
    v7 = tex1Dfetch(texNode7, threadRay->frontid);


    //Ray Bilinear patch intersection
    float2 t0 = make_float2(CUDART_INF_F, 0);
    float2 t1 = make_float2(CUDART_INF_F, 0);
    float2 t2 = make_float2(CUDART_INF_F, 0);
    float2 t3 = make_float2(CUDART_INF_F, 0);
    float2 t4 = make_float2(CUDART_INF_F, 0);
    float2 t5 = make_float2(CUDART_INF_F, 0);

    
    ///res
    /*
    t0.x = Intersect(threadRay, v0, v1, v2, v3, threadRay->frontface == 0 ? 1 : 0).x;
    t1.x = fminf(t, Intersect(threadRay, v4, v5, v6, v7, threadRay->frontface == 1 ? 1 : 0).x);
    t2.x = fminf(t, Intersect(threadRay, v1, v3, v5, v7, threadRay->frontface == 2 ? 1 : 0).x);
    t3.x = fminf(t, Intersect(threadRay, v0, v2, v4, v6, threadRay->frontface == 3 ? 1 : 0).x);
    t4.x = fminf(t, Intersect(threadRay, v2, v3, v6, v7, threadRay->frontface == 5 ? 1 : 0).x);
    t5.x = fminf(t, Intersect(threadRay, v0, v1, v4, v5, threadRay->frontface == 4 ? 1 : 0).x);
    */
    
    //fem
    
    t0.x = Intersect(threadRay, v0, v1, v3, v2, threadRay->frontface == 1 ? 1 : 0).x;
    t1.x = fminf(t, Intersect(threadRay, v5, v4, v6, v7, threadRay->frontface == 0 ? 1 : 0).x);
    t2.x = fminf(t, Intersect(threadRay, v1, v2, v5, v6, threadRay->frontface == 2 ? 1 : 0).x);
    t3.x = fminf(t, Intersect(threadRay, v0, v3, v4, v7, threadRay->frontface == 3 ? 1 : 0).x);
    t4.x = fminf(t, Intersect(threadRay, v2, v3, v6, v7, threadRay->frontface == 5 ? 1 : 0).x);
    t5.x = fminf(t, Intersect(threadRay, v0, v1, v4, v5, threadRay->frontface == 4 ? 1 : 0).x);
    

    if(t0.x < t1.x && t0.x < t2.x && t0.x < t3.x && t0.x < t4.x && t0.x < t5.x){
      t = t0.x;
      //round = hexAdj1.y; //fem
      round = hexAdj1.x; //res
      //round = 0;
      if(constMemory.debug) threadRay->acccolor = make_float4(1, 0, 0, 1);
    }
    if(t1.x < t0.x && t1.x < t2.x && t1.x < t3.x && t1.x < t4.x && t1.x < t5.x){
      t = t1.x;
      //round = hexAdj1.x; //fem
      round = hexAdj1.y; //res
      //round = 0;
      if(constMemory.debug) threadRay->acccolor = make_float4(0, 1, 0, 1);
    }
    if(t2.x < t0.x && t2.x < t1.x && t2.x < t3.x && t2.x < t4.x && t2.x < t5.x){
      t = t2.x;
      //round = hexAdj1.z; //fem
      round = hexAdj1.z; //res
      //round = 0;
      if(constMemory.debug) threadRay->acccolor = make_float4(0, 1, 1, 1);
    }
    if(t3.x < t0.x && t3.x < t1.x && t3.x < t2.x && t3.x < t4.x && t3.x < t5.x){
      t = t3.x;
      //round = hexAdj2.x; //fem
      round = hexAdj2.x; //res
      //round = 0;
      if(constMemory.debug) threadRay->acccolor = make_float4(0, 0, 1, 1);
    }
    if(t4.x < t0.x && t4.x < t1.x && t4.x < t2.x && t4.x < t3.x && t4.x < t5.x){
      t = t4.x;
      //round = hexAdj2.z; //fem
      round = hexAdj2.z; //res
      //round = 0;
      if(constMemory.debug) threadRay->acccolor = make_float4(1, 1, 0, 1);
    }
    if(t5.x < t0.x && t5.x < t1.x && t5.x < t2.x && t5.x < t3.x && t5.x < t4.x){
      t = t5.x;
      //round = hexAdj2.y; //fem
      round = hexAdj2.y; //res
      //round = 0;
      if(constMemory.debug) threadRay->acccolor = make_float4(1, 0, 1, 1);
    }

    if(constMemory.debug) return;

#else

    /*Face 0*/
    if(threadRay->frontface != 0){
      //Triangle 0
      planeEq = tex1Dfetch(texFace0Eq, threadRay->frontid);
      sameDirection = dot(threadRay->dir, planeEq);
      if(sameDirection > 0){
        sameDirection = - dot(planeEq, threadRay->eyepos) / sameDirection;
        if(sameDirection < t){
          t = sameDirection;
          if(constMemory.debug > 0) threadRay->acccolor = make_float4(1, 0, 0, 1);
          //threadRay->accColor = make_float4(0.5, 0.5, 0.5, 1);
          round = hexAdj1.x;
        }
      }
    }

    /*Face 1*/
    if(threadRay->frontface != 1){
      //Triangle 0
      planeEq = tex1Dfetch(texFace1Eq, threadRay->frontid);
      sameDirection = dot(threadRay->dir, planeEq);
      if(sameDirection > 0){
        sameDirection = - dot(planeEq, threadRay->eyepos) / sameDirection;
        if(sameDirection < t){
          t = sameDirection;
          if(constMemory.debug > 0) threadRay->acccolor = make_float4(0, 1, 0, 1);
          //threadRay->accColor = make_float4(0.5, 0.5, 0.5, 1);
          round = hexAdj1.y;
        }
      }
    }

    /*Face 2*/
    if(threadRay->frontface != 2){
      //Triangle 0
      planeEq = tex1Dfetch(texFace2Eq, threadRay->frontid);
      sameDirection = dot(threadRay->dir, planeEq);
      if(sameDirection > 0){
        sameDirection = - dot(planeEq, threadRay->eyepos) / sameDirection;
        if(sameDirection < t){
          t = sameDirection;
          if(constMemory.debug > 0) threadRay->acccolor = make_float4(0, 0, 1, 1);
          //threadRay->accColor = make_float4(0.5, 0.5, 0.5, 1);
          round = hexAdj1.z;
        }
      }
    }

    /*Face 3*/
    if(threadRay->frontface != 3){
      //Triangle 0
      planeEq = tex1Dfetch(texFace3Eq, threadRay->frontid);
      sameDirection = dot(threadRay->dir, planeEq);
      if(sameDirection > 0){
        sameDirection = - dot(planeEq, threadRay->eyepos) / sameDirection;
        if(sameDirection < t){
          t = sameDirection;
          if(constMemory.debug > 0) threadRay->acccolor = make_float4(1, 1, 0, 1);
          //threadRay->accColor = make_float4(0.5, 0.5, 0.5, 1);
          round = hexAdj2.x;
        }
      }
    }

    /*Face 4*/
    if(threadRay->frontface != 4){
      //Triangle 0
      planeEq = tex1Dfetch(texFace4Eq, threadRay->frontid);
      sameDirection = dot(threadRay->dir, planeEq);
      if(sameDirection > 0){
        sameDirection = - dot(planeEq, threadRay->eyepos) / sameDirection;
        if(sameDirection < t){
          t = sameDirection;
          if(constMemory.debug > 0) threadRay->acccolor = make_float4(1, 0, 1, 1);
          //threadRay->accColor = make_float4(0.5, 0.5, 0.5, 1);
          round = hexAdj2.y;
        }
      }
    }

    /*Face 5*/
    if(threadRay->frontface != 5){
      //Triangle 0
      planeEq = tex1Dfetch(texFace5Eq, threadRay->frontid);
      sameDirection = dot(threadRay->dir, planeEq);
      if(sameDirection > 0){
        sameDirection = - dot(planeEq, threadRay->eyepos) / sameDirection;
        if(sameDirection < t){
          t = sameDirection;
          if(constMemory.debug > 0) threadRay->acccolor = make_float4(0, 1, 1, 1);
          //threadRay->accColor = make_float4(0.5, 0.5, 0.5, 1);
          round = hexAdj2.z;
        }
      }
    }




#endif
#else
#ifdef CUDARC_PLUCKER

    v0 = tex1Dfetch(texNode0, threadRay->frontid);
    v1 = tex1Dfetch(texNode1, threadRay->frontid);
    v2 = tex1Dfetch(texNode2, threadRay->frontid);
    v3 = tex1Dfetch(texNode3, threadRay->frontid);

    v02 = (v0 - v2);
    q02 = cross(v02, v0);
    ps02 = permuted_inner_produtct(v02, q02, threadRay->dir, ray);

    v32 = (v3 - v2);
    q32 = cross(v32, v2);
    ps32 = permuted_inner_produtct(v32, q32, threadRay->dir, ray);

    v03 = (v0 - v3);
    q03 = cross(v03, v0);
    ps03 = permuted_inner_produtct(v03, q03, threadRay->dir, ray);

    v13 = (v1 - v3);
    q13 = cross(v13, v1);
    ps13 = permuted_inner_produtct(v13, q13, threadRay->dir, ray);

    v01 = (v0 - v1);
    q01 = cross(v01, v0);
    ps01 = permuted_inner_produtct(v01, q01, threadRay->dir, ray);

    v21 = (v2 - v1);
    q21 = cross(v21, v1);
    ps21 = permuted_inner_produtct(v21, q21, threadRay->dir, ray);


    //Plucker tests
    if(threadRay->frontface == 0.0f){
      //Face 0: 1, 2, 3
      /*
      v0 = tex1Dfetch(texNode0, threadRay->frontid);

      v01 = (v0 - v1);
      q01 = cross(v01, v0);
      ps01 = permuted_inner_produtct(v01, q01, threadRay->dir, ray);

      v02 = (v0 - v2);
      q02 = cross(v02, v0);
      ps02 = permuted_inner_produtct(v02, q02, threadRay->dir, ray);

      v03 = (v0 - v3);
      q03 = cross(v03, v0);
      ps03 = permuted_inner_produtct(v03, q03, threadRay->dir, ray);

      v21 = - v21;
      v13 = - v13;
      v32 = - v32;
      */
      //Test against faces 1, 2, 3
      if((-ps32 <= 0 && -ps03 <= 0 && ps02 <= 0)){
        float3 u = make_float3(-ps32, -ps03, ps02) / (-ps32 -ps03 + ps02);
        if(constMemory.debug) threadRay->acccolor = make_float4(0, 1, 0, 1);
        round = tetAdj.y;
        point = u.x * v0 +  u.y * v2 + u.z * v3;
        point.w = 1.0f;
      }
      else{
        if((-ps13 <= 0 && -ps01 <= 0 && ps03 <= 0)){
          float3 u = make_float3(-ps13, -ps01, ps03) / (-ps13 -ps01 + ps03);
          if(constMemory.debug) threadRay->acccolor = make_float4(0, 0, 1, 1);
          round = tetAdj.z;
          point = u.x * v0 +  u.y * v3 + u.z * v1;
          point.w = 1.0f;
        }
        else{
          if((-ps21 <= 0 && -ps02 <= 0 && ps01 <= 0)){
            float3 u = make_float3(-ps21, -ps02, ps01) / (-ps21 -ps02 + ps01);
            if(constMemory.debug) threadRay->acccolor = make_float4(1, 0, 1, 1);
            round = tetAdj.w;
            point = u.x * v0 +  u.y * v1 + u.z * v2;
            point.w = 1.0f;
          }
        } 
      }
    }
    else if(threadRay->frontface == 1.0f){
      //Face 1: 2, 0, 3

      /*
      v1 = tex1Dfetch(texNode1, threadRay->frontid);

      v13 = (v1 - v3);
      q13 = cross(v13, v1);
      ps13 = permuted_inner_produtct(v13, q13, threadRay->dir, ray);

      v01 = (v0 - v1);
      q01 = cross(v01, v0);
      ps01 = permuted_inner_produtct(v01, q01, threadRay->dir, ray);

      v21 = (v2 - v1);
      q21 = cross(v21, v1);
      ps21 = permuted_inner_produtct(v21, q21, threadRay->dir, ray);

      v02 = - v02;
      v03 = - v03;
      v32 = - v32;
      */
      //Test against faces 0, 2, 3
      if((ps13 <= 0 && ps32 <= 0 && ps21 <= 0)){
        float3 u = make_float3(ps13, ps32, ps21) / (ps13 + ps32 + ps21);
        if(constMemory.debug) threadRay->acccolor = make_float4(1, 0, 0, 1);
        round = tetAdj.x;
        point = u.x * v2 +  u.y * v1 + u.z * v3;
        point.w = 1.0f;
      }
      else{
        if((-ps13 <= 0 && -ps01 <= 0 && ps03 <= 0)){
          float3 u = make_float3(-ps13, -ps01, ps03) / (-ps13 -ps01 + ps03);
          if(constMemory.debug) threadRay->acccolor = make_float4(0, 0, 1, 1);
          round = tetAdj.z;
          point = u.x * v0 +  u.y * v3 + u.z * v1;
          point.w = 1.0f;
        }
        else{
          if((-ps21 <= 0 && -ps02 <= 0 && ps01 <= 0)){
            float3 u = make_float3(-ps21, -ps02, ps01) / (-ps21 -ps02 + ps01);
            if(constMemory.debug) threadRay->acccolor = make_float4(1, 0, 1, 1);
            round = tetAdj.w;
            point = u.x * v0 +  u.y * v1 + u.z * v2;
            point.w = 1.0f;
          }
        } 
      }
    }
    else if(threadRay->frontface == 2.0f){
      //Face 2: 3, 0, 1
      /*
      v2 = tex1Dfetch(texNode2, threadRay->frontid);

      v02 = (v0 - v2);
      q02 = cross(v02, v0);
      ps02 = permuted_inner_produtct(v02, q02, threadRay->dir, ray);

      v32 = (v3 - v2);
      q32 = cross(v32, v2);
      ps32 = permuted_inner_produtct(v32, q32, threadRay->dir, ray);

      v21 = (v2 - v1);
      q21 = cross(v21, v1);
      ps21 = permuted_inner_produtct(v21, q21, threadRay->dir, ray);

      v01 = - v01;
      v03 = - v03;
      v13 = - v13;
      */
      //Test against faces 0, 1, 3
      if((ps13 <= 0 && ps32 <= 0 && ps21 <= 0)){
        float3 u = make_float3(ps13, ps32, ps21) / (ps13 + ps32 + ps21);
        if(constMemory.debug) threadRay->acccolor = make_float4(1, 0, 0, 1);
        round = tetAdj.x;
        point = u.x * v2 +  u.y * v1 + u.z * v3;
        point.w = 1.0f;
      }
      else{
        if((-ps32 <= 0 && -ps03 <= 0 && ps02<= 0)){
          float3 u = make_float3(-ps32, -ps03, ps02) / (-ps32 -ps03 + ps02);
          if(constMemory.debug) threadRay->acccolor = make_float4(0, 1, 0, 1);
          round = tetAdj.y;
          point = u.x * v0 +  u.y * v2 + u.z * v3;
          point.w = 1.0f;
        }
        else{
          if((-ps21 <= 0 && -ps02 <= 0 && ps01 <= 0)){
            float3 u = make_float3(-ps21, -ps02, ps01) / (-ps21 -ps02 + ps01);
            if(constMemory.debug) threadRay->acccolor = make_float4(1, 0, 1, 1);
            round = tetAdj.w;
            point = u.x * v0 +  u.y * v1 + u.z * v2;
            point.w = 1.0f;
          }
        } 
      }
    }
    else if(threadRay->frontface == 3.0f){
      //Face 3: 1, 0, 2

      //threadRay->acccolor = make_float4(1, 0, 0, 1);
      //return;
      /*
      v3 = tex1Dfetch(texNode3, threadRay->frontid);

      v32 = (v3 - v2);
      q32 = cross(v32, v2);
      ps32 = permuted_inner_produtct(v32, q32, threadRay->dir, ray);

      v03 = (v0 - v3);
      q03 = cross(v03, v0);
      ps03 = permuted_inner_produtct(v03, q03, threadRay->dir, ray);

      v13 = (v1 - v3);
      q13 = cross(v13, v1);
      ps13 = permuted_inner_produtct(v13, q13, threadRay->dir, ray);

      v01 = - v01;
      v02 = - v02;
      v21 = - v21;
      */
      //Test against faces 0, 1, 2
      if((ps13 <= 0 && ps32 <= 0 && ps21 <= 0)){
        float3 u = make_float3(ps13, ps32, ps21) / (ps13 + ps32 + ps21);
        if(constMemory.debug) threadRay->acccolor = make_float4(1, 0, 0, 1);
        round = tetAdj.x;
        point = u.x * v2 +  u.y * v1 + u.z * v3;
        point.w = 1.0f;
      }
      else{
        if((-ps32 <= 0 && -ps03 <= 0 && ps02 <= 0)){
          float3 u = make_float3(-ps32, -ps03, ps02) / (-ps32 -ps03 + ps02);
          if(constMemory.debug) threadRay->acccolor = make_float4(0, 1, 0, 1);
          round = tetAdj.y;
          point = u.x * v0 +  u.y * v2 + u.z * v3;
          point.w = 1.0f;
        }
        else{
          if((-ps13 <= 0 && -ps01 <= 0 && ps03 <= 0)){
            float3 u = make_float3(-ps13, -ps01, ps03) / (-ps13 -ps01 + ps03);
            if(constMemory.debug) threadRay->acccolor = make_float4(0, 0, 1, 1);
            round = tetAdj.z;
            point = u.x * v0 +  u.y * v3 + u.z * v1;
            point.w = 1.0f;
          }
        } 
      }
    }


    t = length(point - threadRay->eyepos);
    //threadRay->acccolor = make_float4(t, t, t, 1);

    //threadRay->acccolor = make_float4(normalize(point - eyePos).x, normalize(point - eyePos).y, normalize(point - eyePos).z, 1);
    //return;

#else

    /*Face 0*/
    if(threadRay->frontface != 0){
      //Triangle 0
      planeEq = tex1Dfetch(texFace0Eq, threadRay->frontid);
      sameDirection = dot(threadRay->dir, planeEq);
      if(sameDirection > 0){
        sameDirection = - dot(planeEq, threadRay->eyepos) / sameDirection;
        if(sameDirection < t){
          t = sameDirection;
          if(constMemory.debug > 0) threadRay->acccolor = make_float4(1, 0, 0, 1);
          //threadRay->accColor = make_float4(0.5, 0.5, 0.5, 1);
          round = tetAdj.x;
        }
      }
    }

    /*Face 1*/
    if(threadRay->frontface != 1){
      //Triangle 0
      planeEq = tex1Dfetch(texFace1Eq, threadRay->frontid);
      sameDirection = dot(threadRay->dir, planeEq);
      if(sameDirection > 0){
        sameDirection = - dot(planeEq, threadRay->eyepos) / sameDirection;
        if(sameDirection < t){
          t = sameDirection;
          if(constMemory.debug > 0) threadRay->acccolor = make_float4(0, 1, 0, 1);
          //threadRay->accColor = make_float4(0.5, 0.5, 0.5, 1);
          round = tetAdj.y;
        }
      }
    }

    /*Face 2*/
    if(threadRay->frontface != 2){
      //Triangle 0
      planeEq = tex1Dfetch(texFace2Eq, threadRay->frontid);
      sameDirection = dot(threadRay->dir, planeEq);
      if(sameDirection > 0){
        sameDirection = - dot(planeEq, threadRay->eyepos) / sameDirection;
        if(sameDirection < t){
          t = sameDirection;
          if(constMemory.debug > 0) threadRay->acccolor = make_float4(0, 0, 1, 1);
          //threadRay->accColor = make_float4(0.5, 0.5, 0.5, 1);
          round = tetAdj.z;
        }
      }
    }

    /*Face 3*/
    if(threadRay->frontface != 3){
      //Triangle 0
      planeEq = tex1Dfetch(texFace3Eq, threadRay->frontid);
      sameDirection = dot(threadRay->dir, planeEq);
      if(sameDirection > 0){
        sameDirection = - dot(planeEq, threadRay->eyepos) / sameDirection;
        if(sameDirection < t){
          t = sameDirection;
          if(constMemory.debug > 0) threadRay->acccolor = make_float4(1, 1, 0, 1);
          //threadRay->accColor = make_float4(0.5, 0.5, 0.5, 1);
          round = tetAdj.w;
        }
      }
    }
    /*
    float4 tempdir;
    if(t >= 0 && t < CUDART_INF_F){
    tempdir = (threadRay->eyepos + threadRay->dir * t) - threadRay->eyepos;
    }
    else{
    tempdir = make_float4(0, 0, 0, 0);
    }
    tempdir = normalize(tempdir);
    threadRay->acccolor = make_float4(tempdir.x, tempdir.y, tempdir.z, 1);
    return;
    */
#endif
#endif

    int rounded = floor(round + 0.5f);
#ifdef CUDARC_HEX
    backid = rounded / 6; 
    backfaceid = rounded % 6; 
#else
    backid = rounded / 4; 
    backfaceid = rounded % 4; 
#endif

    //return;


    //Round
    t = fmaxf(t, threadRay->t);


    float tetraBackScalar;
    tetraBackScalar = FindScalar(threadRay, t);

    if(constMemory.interpoltype == Step){
      float frontt = threadRay->t;
      float step = (t - frontt) / constMemory.numsteps;
      float frontscalar = threadRay->frontscalar;
      float backscalar = tetraBackScalar;
      for(int i=0; i<constMemory.numsteps; i++){
        IntegrateRayConst(threadRay, threadRay->eyepos, step / constMemory.maxedge, frontscalar, backscalar);
        frontt += step; 

        frontscalar = backscalar;
        backscalar = FindScalar(threadRay, frontt);

        if(threadRay->acccolor.w > 0.99)
          break;
      }
    }
    else{
      float isocpscalar = 3.0f;
      float volcpscalar = 3.0f;

      if(constMemory.isosurface)
        isocpscalar = FindControlPoint(threadRay, tetraBackScalar, tex1D(texIsoControlPoints, threadRay->frontscalar));

      if(constMemory.volumetric)
        volcpscalar = FindControlPoint(threadRay, tetraBackScalar, tex1D(texVolumetricControlPoints, threadRay->frontscalar));
      

      //Find if tetra contains cp
      float diffisocpfront = fabs(isocpscalar - threadRay->frontscalar);
      float diffvolcpfront = fabs(volcpscalar - threadRay->frontscalar);
      float diffbackfront = fabs(tetraBackScalar - threadRay->frontscalar);

      float diffcpfront = 3.0f;
      float cpscalar = 3.0f;
      if(diffisocpfront - diffvolcpfront < 1e-6){
        diffcpfront = diffisocpfront;
        cpscalar = isocpscalar;
      }
      else{
        diffcpfront = diffvolcpfront;
        cpscalar = volcpscalar;
      }

      if(diffcpfront - diffbackfront < 1e-6){
        //Integrate between front and iso, but dont traverse to the next tetra
        backid = threadRay->frontid;
        backfaceid = threadRay->frontface;
        tetraBackScalar = cpscalar;
        
        t = FindIntegrationStep(threadRay, t, diffcpfront, diffbackfront);
        
        /*
        float c0 = threadRay->currentelem.interpolfunc1.w;
        float c1 = threadRay->currentelem.interpolfunc0.x;
        float c2 = threadRay->currentelem.interpolfunc0.y;
        float c3 = threadRay->currentelem.interpolfunc0.z;
        float c4 = threadRay->currentelem.interpolfunc0.w;
        float c5 = threadRay->currentelem.interpolfunc1.x;
        float c6 = threadRay->currentelem.interpolfunc1.y;
        float c7 = threadRay->currentelem.interpolfunc1.z;

        float ox = threadRay->eyepos.x;
        float oy = threadRay->eyepos.y;
        float oz = threadRay->eyepos.z;

        float dx = threadRay->dir.x;
        float dy = threadRay->dir.y;
        float dz = threadRay->dir.z;

        float a = c0 + c1*ox + c2*oy + c4*ox*oy + c3*oz + c6*ox*oz + c5*oy*oz + c7*ox*oy*oz;
        float b = c1*dx + c2*dy + c3*dz + c4*dy*ox + c6*dz*ox + c4*dx*oy + c5*dz*oy + c7*dz*ox*oy + c6*dx*oz + c5*dy*oz + c7*dy*ox*oz + c7*dx*oy*oz;
        float c = c4*dx*dy + c6*dx*dz + c5*dy*dz + c7*dy*dz*ox + c7*dx*dz*oy + c7*dx*dy*oz;
        float d = c7*dx*dy*dz - cpscalar;


        float delta = 18.0f * a * b * c * d + 4.0f * b * b * b * d + b * b + c * c - 4 * a * c * c * c - 27.0f * a * a * d * d;

        if(delta > 0.0f)
          threadRay->acccolor = make_float4(1, 0, 0, 1);
        else if(delta == 0)
          threadRay->acccolor = make_float4(0, 1, 0, 1);
        else
          threadRay->acccolor = make_float4(0, 0, 1, 1);
        return;
        */
      }
      

      //Volumetric
      if(constMemory.volumetric > 0){
        float diff = t - threadRay->t;
        if(constMemory.debug == 0 && diff > 0){

          if(constMemory.interpoltype == Const){
            IntegrateRayConst(threadRay, threadRay->eyepos, diff / constMemory.maxedge, tetraBackScalar, threadRay->frontscalar);
          }
          else{
            IntegrateRayLinear(threadRay, threadRay->eyepos, diff / constMemory.maxedge, tetraBackScalar, threadRay->frontscalar);
          }
        }
      }
      
      //Isosurface
      if(constMemory.isosurface > 0 && backid == threadRay->frontid && cpscalar == isocpscalar){

#ifdef CUDARC_GRADIENT_PERVERTEX
        //Initialize barycentric interpolation
        float4 gradv0   = tex1Dfetch(texGrad0, threadRay->frontid);
        float4 gradv1   = tex1Dfetch(texGrad1, threadRay->frontid);
        float4 gradv2   = tex1Dfetch(texGrad2, threadRay->frontid);
        float4 gradv3   = tex1Dfetch(texGrad3, threadRay->frontid);
        float4 gradient = (gradv0 + gradv1 + gradv2 + gradv3)/4.0;
        float3 N = normalize(make_float3(gradient.x,gradient.y,gradient.z));
#else
        float3 N = normalize(make_float3(threadRay->currentelem.interpolfunc0));
#endif

        float4 color = tex1D(texIsoColorScale, tetraBackScalar);
        float3 L = normalize(make_float3(- threadRay->t * threadRay->dir));
        color.x *= abs(dot(N, L));
        color.y *= abs(dot(N, L));
        color.z *= abs(dot(N, L));
        color.x *= color.w; color.y *= color.w; color.z *= color.w;
        
        threadRay->acccolor += (1.0f - threadRay->acccolor.w) * color;
        //threadRay->acccolor = make_float4(make_float3(color), 1);
        //threadRay->acccolor = make_float4(1, 0, 0, 1);
      }
    }

    //threadRay->acccolor.w = 1;

    //Traverse
    threadRay->frontid = backid;
    threadRay->frontface = backfaceid;
    threadRay->frontscalar = tetraBackScalar;
    threadRay->t = t;

    threadRay->currentelem.interpolfunc0 = tex1Dfetch(texInterpolFunc0, threadRay->frontid);
#ifdef CUDARC_HEX
    threadRay->currentelem.interpolfunc1 = tex1Dfetch(texInterpolFunc1, threadRay->frontid);
    hexAdj1 = tex1Dfetch(texAdj0, threadRay->frontid);
    hexAdj2 = tex1Dfetch(texAdj1, threadRay->frontid);
#else
    tetAdj = tex1Dfetch(texAdj0, threadRay->frontid);
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
extern "C" void init(GLuint p_handleTexIntersect, GLuint p_handlePboOutput){




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
  //CUDA_SAFE_CALL(cudaMalloc((void**)&dev_debug, sizex * sizey * sizeof(float4)));

}

/**
* CUDA callback (device)
*/
__global__ void Run(int depthPeelPass, float4 eyePos, float4* dev_outputData){

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int offset = x + y * blockDim.x * gridDim.x;

  Ray threadRay = Initialize(x, y, offset, eyePos);

  if(depthPeelPass > 0)
    threadRay.acccolor = dev_outputData[offset];
  

  if(threadRay.frontid != 0)
    Traverse(x, y, offset, &threadRay);


  dev_outputData[offset] = threadRay.acccolor;
}

/**
* CUDA callback (host)
*/
extern "C" void run(float* p_kernelTime, float* p_overheadTime, int depthPeelPass, float* p_eyePos, int handleTexIntersect){

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
  texVolumetricControlPoints.filterMode = cudaFilterModeLinear;
  texVolumetricControlPoints.normalized = true;
  CUDA_SAFE_CALL(cudaBindTextureToArray(texVolumetricControlPoints, dev_volcontrolpoints));

  texIsoControlPoints.addressMode[0] = cudaAddressModeClamp;
  texIsoControlPoints.filterMode = cudaFilterModeLinear;
  texIsoControlPoints.normalized = true;
  CUDA_SAFE_CALL(cudaBindTextureToArray(texIsoControlPoints, dev_isocontrolpoints));


  texZetaPsiGamma.addressMode[0] = cudaAddressModeClamp;
  texZetaPsiGamma.addressMode[1] = cudaAddressModeClamp;
  texZetaPsiGamma.filterMode = cudaFilterModeLinear;
  texZetaPsiGamma.normalized = true;
  CUDA_SAFE_CALL(cudaBindTextureToArray(texZetaPsiGamma, dev_zetaPsiGamma));


  //Camera settings
  float4 eyePos = make_float4(p_eyePos[0], p_eyePos[1], p_eyePos[2], 1);

#ifdef CUDARC_TIME
  CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop));
  CUDA_SAFE_CALL(cudaEventElapsedTime(p_overheadTime, start, stop));

  CUDA_SAFE_CALL(cudaEventRecord(start, 0));
#endif

  //Kernel call
  Run<<<grids, threads>>>(depthPeelPass, eyePos, dev_outputData);


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
extern "C" void createGPUAdjTex(int index, int size, float* data){
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
extern "C" void createGPUGradientVertexTex(int ni, int size, float* data)
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
extern "C" void createGPUCollisionTex(int ni, int size, float* data){
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
extern "C" void createGPUCollisionTex(int fi, int size, float* data){
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
extern "C" void createGPUInterpolFuncTex(int index, int size, float* data){
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
extern "C" void createGPUColorScaleTex(int numValues, int size, float* volcolorscaledata, float* isocolorscale){
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
extern "C" void createGPUVolControlPointsTex(int numValues, int size, float* data){
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
  CUDA_SAFE_CALL(cudaMallocArray(&dev_volcontrolpoints, &channelDesc, numValues));
  CUDA_SAFE_CALL(cudaMemcpyToArray(dev_volcontrolpoints, 0, 0, data, size, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaBindTextureToArray(texVolumetricColorScale, dev_volcontrolpoints, channelDesc)); 

#ifdef CUDARC_VERBOSE
  printf("Control points to CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
#endif
}

extern "C" void createGPUIsoControlPointsTex(int numValues, int size, float* data){
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
extern "C" void createGPUZetaPsiGammaTex(int numValues, int size, float* data){
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
extern "C" void deleteGPUTextures(int numAdjTex, int numInterpolFuncTex){

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
extern "C" void printInfoGPUMemory(){
  unsigned int free, total;
  cuMemGetInfo(&free, &total);
  printf("#GPU Mem Info: Free = %d (%f), Total = %d\n", free, (float)free/(float)total, total);
}


/**
* Set const memory values
*/
extern "C" void update(int p_blocksizex, int p_blocksizey, int p_winsizex, int p_winsizey, bool p_debug, float p_maxedge, int p_interpoltype, int p_numsteps, int p_numtraverses, int p_numelem, bool p_isosurface, bool p_volumetric){

  grids = dim3(p_winsizex / p_blocksizex, p_winsizey / p_blocksizey);
  threads = dim3(p_blocksizex, p_blocksizey);

  ConstMemory *tempConstMemory = (ConstMemory*)malloc( sizeof(ConstMemory));
  tempConstMemory->numTets = p_numelem;
  tempConstMemory->screenSize = make_float2(p_winsizex, p_winsizey);
  tempConstMemory->maxedge = p_maxedge;
  tempConstMemory->interpoltype = p_interpoltype;
  tempConstMemory->numsteps = p_numsteps;
  tempConstMemory->numtraverses = p_numtraverses;
  tempConstMemory->debug = p_debug;
  tempConstMemory->isosurface = p_isosurface;
  tempConstMemory->volumetric = p_volumetric;

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(constMemory, tempConstMemory, sizeof(ConstMemory)));
  delete tempConstMemory;
}
