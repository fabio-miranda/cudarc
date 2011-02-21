/**
* Fabio Markus Miranda
* fmiranda@tecgraf.puc-rio.br
* fabiom@gmail.com
* Dec 2010
* 
* interpolfunc.cpp: compute the interpolation function of the elements, using SVD
*/

#include "interpolfunc.h"

#include <alg/vector.h>
#include <alg/matrix.h>

#include <limits>

void ComputeGradientLeastSquares(int numvertices, float* positions, float* scalars, float* gradient){

  long double** A = new long double*[numvertices];
  long double** b = new long double*[numvertices];
  for(int i=0; i<numvertices; i++){
    A[i] = new long double[numvertices];
    b[i] = new long double[numvertices];
  }

  int count = 0;
  for(int i=0; i<numvertices; i++){
    //if(i != centralvertex){
      
      long double x = positions[3*i+0];
      long double y = positions[3*i+1];
      long double z = positions[3*i+2];

      A[count][0] = x;
      A[count][1] = y;
      A[count][2] = z;
      if(numvertices > 4){
        A[count][3] = x * y;
        A[count][4] = x * z;
        A[count][5] = y * z;
        A[count][6] = x * y * z;
        A[count][7] = 1.0;

        b[count][0] = scalars[i];
        b[count][1] = 0;
        b[count][2] = 0;
        b[count][3] = 0;
        b[count][4] = 0;
        b[count][5] = 0;
        b[count][6] = 0;
        b[count][7] = 0;
      }
      else{
        A[count][3] = 1.0;

        b[count][0] = scalars[i];
        b[count][1] = 0;
        b[count][2] = 0;
        b[count][3] = 0;
      }

      count++;
   // }
  }

  if(!gradient)
    gradient = new float[numvertices];


  long double* result = new long double[numvertices];
  if(LeastSquares(numvertices, numvertices, A, b, result)){
    for(int i=0; i<numvertices; i++){
      //if(fabs(result[i]) < 1e-8)
        //gradient[i] = 0.0f;
      //else
        gradient[i] = (float)result[i];
    }
  }

}

int LeastSquares(int numlines, int numcols, long double** A, long double** b, long double* x){
  

  long double* W = new long double[numcols];
  long double** V = new long double*[numcols];
  for(int i=0; i<numcols; i++)
    V[i] = new long double[numlines];

  //long double** trans = Transpose(size+1, 8+1, U);
  svdcmp(A, numlines, numcols, W, V);

  //Calculate inv(A) = V * inv(W) * Ut
  //inv(W)
  long double** invW = new long double*[numlines];
  for(int i=0; i<numlines; i++){
    invW[i] = new long double[numcols];
    for(int j=0; j<numcols; j++)
      invW[i][j] = 0;
  }

  //Calculate tolerance first
  long double norm = 0;
  for(int i=0; i<numlines; i++){
    if(W[i] > norm)
      norm = W[i];
  }

  long double tolerance = norm * numlines * std::numeric_limits<long double>::epsilon();
  //long double tolerance = 0;
  //long double tolerance = 10e-5;
  for(int i=0; i<numlines; i++){
    if(W[i] > tolerance)
      invW[i][i] = 1.0 / W[i];
    else
      invW[i][i] = 0;
  }

  //Ut
  long double** Ut = Transpose(numlines, numcols, A);

  //inv(A)
  long double** aux = Multiply(numcols, numcols, V, numcols, numcols, invW);
  long double** invA = Multiply(numcols, numlines, aux, numcols, numlines, Ut);

  //(inv(AtA) * At)
  //long double** invAtAAt = Multiply(8, 8, invAtA, 8, size, At);
  long double** result = Multiply(numcols, numlines, invA, numlines, 1, b);

  for(int i=0; i<numlines; i++)
    x[i] = result[i][0];

  for(int i=0; i<numcols; i++){
    delete [] result[i];
    delete [] invA[i];
    delete [] Ut[i];
    delete [] invW[i];
    delete [] V[i];
    delete [] aux[i];
  }
  delete [] result;
  delete [] invA;
  delete [] Ut;
  delete [] invW;
  delete [] W;
  delete [] V;
  delete [] aux;

  return 1;


}

long double** Transpose(int a_m, int a_n, long double** a){
  long double** result = new long double*[a_n];
  for(int i=0; i<a_n; i++)
    result[i] = new long double[a_m];

  for(int i=0; i<a_n; i++)
    for(int j=0; j<a_m; j++)
      result[i][j] = a[j][i];

  return result;
}

long double** Multiply(int a_m, int a_n, long double** a, int b_m, int b_n, long double** b){

  long double** result = new long double*[a_m];
  for(int i=0; i<a_m; i++)
    result[i] = new long double[b_n];


  for(int i=0; i<a_m; i++)
    for(int j=0; j<b_n; j++)
      result[i][j] = 0;
  

  for(int i=0; i<a_m; i++)
    for(int j=0; j<b_n; j++)
      for(int k=0; k<a_n; k++)
        result[i][j] += a[i][k] * b[k][j];

  return result;
}

/**
* From Numerical Recipes in C: The Art of Scientific Computing
*/

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
static long double maxarg1,maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
  (maxarg1) : (maxarg2))
static int iminarg1,iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
  (iminarg1) : (iminarg2))
#define NR_END 1
#define FREE_ARG char*
static long double sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)
void free_vector(long double *v, long nl, long nh)
/* free a long double vector allocated with vector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}
long double *vector(long nl, long nh)
/* allocate a long double vector with subscript range v[nl..nh] */
{
  long double *v;

  v=(long double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(long double)));
  if (!v){
    printf("Error: Allocation failure in vector()\n");
    exit(1);
  }
  return v-nl+NR_END;
}
long double pythag(long double a, long double b)
{
  long double absa,absb;
  absa=fabs(a);
  absb=fabs(b);
  if (absa > absb) return absa*sqrt(1.0+SQR(absb/absa));
  else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+SQR(absa/absb)));
}


void svdcmp(long double **a, int m, int n, long double w[], long double **v)
{
  long double pythag(long double a, long double b);
  int flag,i,its,j,jj,k,l,nm;
  long double anorm,c,f,g,h,s,scale,x,y,z,*rv1;

  rv1=vector(1,n);
  g=scale=anorm=0.0;
  for (i=1;i<=n;i++) {
    l=i+1;
    rv1[i-1]=scale*g;
    g=s=scale=0.0;
    if (i <= m) {
      for (k=i;k<=m;k++) scale += fabs(a[k-1][i-1]);
      if (scale) {
        for (k=i;k<=m;k++) {
          a[k-1][i-1] /= scale;
          s += a[k-1][i-1]*a[k-1][i-1];
        }
        f=a[i-1][i-1];
        g = -SIGN(sqrt(s),f);
        h=f*g-s;
        a[i-1][i-1]=f-g;
        for (j=l;j<=n;j++) {
          for (s=0.0,k=i;k<=m;k++) s += a[k-1][i-1]*a[k-1][j-1];
          f=s/h;
          for (k=i;k<=m;k++) a[k-1][j-1] += f*a[k-1][i-1];
        }
        for (k=i;k<=m;k++) a[k-1][i-1] *= scale;
      }
    }
    w[i-1]=scale *g;
    g=s=scale=0.0;
    if (i <= m && i != n) {
      for (k=l;k<=n;k++) scale += fabs(a[i-1][k-1]);
      if (scale) {
        for (k=l;k<=n;k++) {
          a[i-1][k-1] /= scale;
          s += a[i-1][k-1]*a[i-1][k-1];
        }
        f=a[i-1][l-1];
        g = -SIGN(sqrt(s),f);
        h=f*g-s;
        a[i-1][l-1]=f-g;
        for (k=l;k<=n;k++) rv1[k-1]=a[i-1][k-1]/h;
        for (j=l;j<=m;j++) {
          for (s=0.0,k=l;k<=n;k++) s += a[j-1][k-1]*a[i-1][k-1];
          for (k=l;k<=n;k++) a[j-1][k-1] += s*rv1[k-1];
        }
        for (k=l;k<=n;k++) a[i-1][k-1] *= scale;
      }
    }
    anorm=FMAX(anorm,(fabs(w[i-1])+fabs(rv1[i-1])));
  }
  for (i=n;i>=1;i--) {
    if (i < n) {
      if (g) {
        for (j=l;j<=n;j++)
          v[j-1][i-1]=(a[i-1][j-1]/a[i-1][l-1])/g;
        for (j=l;j<=n;j++) {
          for (s=0.0,k=l;k<=n;k++) s += a[i-1][k-1]*v[k-1][j-1];
          for (k=l;k<=n;k++) v[k-1][j-1] += s*v[k-1][i-1];
        }
      }
      for (j=l;j<=n;j++) v[i-1][j-1]=v[j-1][i-1]=0.0;
    }
    v[i-1][i-1]=1.0;
    g=rv1[i-1];
    l=i;
  }
  for (i=IMIN(m,n);i>=1;i--) {
    l=i+1;
    g=w[i-1];
    for (j=l;j<=n;j++) a[i-1][j-1]=0.0;
    if (g) {
      g=1.0/g;
      for (j=l;j<=n;j++) {
        for (s=0.0,k=l;k<=m;k++) s += a[k-1][i-1]*a[k-1][j-1];
        f=(s/a[i-1][i-1])*g;
        for (k=i;k<=m;k++) a[k-1][j-1] += f*a[k-1][i-1];
      }
      for (j=i;j<=m;j++) a[j-1][i-1] *= g;
    } else for (j=i;j<=m;j++) a[j-1][i-1]=0.0;
    ++a[i-1][i-1];
  }
  for (k=n;k>=1;k--) {
    for (its=1;its<=30;its++) {
      flag=1;
      for (l=k;l>=1;l--) {
        nm=l-1;
        if ((long double)(fabs(rv1[l-1])+anorm) == anorm) {
          flag=0;
          break;
        }
        if ((long double)(fabs(w[nm-1])+anorm) == anorm) break;
      }
      if (flag) {
        c=0.0;
        s=1.0;
        for (i=l;i<=k;i++) {
          f=s*rv1[i-1];
          rv1[i-1]=c*rv1[i-1];
          if ((long double)(fabs(f)+anorm) == anorm) break;
          g=w[i-1];
          h=pythag(f,g);
          w[i-1]=h;
          h=1.0/h;
          c=g*h;
          s = -f*h;
          for (j=1;j<=m;j++) {
            y=a[j-1][nm-1];
            z=a[j-1][i-1];
            a[j-1][nm-1]=y*c+z*s;
            a[j-1][i-1]=z*c-y*s;
          }
        }
      }
      z=w[k-1];
      if (l == k) {
        if (z < 0.0) {
          w[k-1] = -z;
          for (j=1;j<=n;j++) v[j-1][k-1] = -v[j-1][k-1];
        }
        break;
      }
      if (its == 30){
        printf("Error: No convergence in 30 svdcmp iterations\n");
        exit(1);
      }
      x=w[l-1];
      nm=k-1;
      y=w[nm-1];
      g=rv1[nm-1];
      h=rv1[k-1];
      f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
      g=pythag(f,1.0);
      f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
      c=s=1.0;
      for (j=l;j<=nm;j++) {
        i=j+1;
        g=rv1[i-1];
        y=w[i-1];
        h=s*g;
        g=c*g;
        z=pythag(f,h);
        rv1[j-1]=z;
        c=f/z;
        s=h/z;
        f=x*c+g*s;
        g = g*c-x*s;
        h=y*s;
        y *= c;
        for (jj=1;jj<=n;jj++) {
          x=v[jj-1][j-1];
          z=v[jj-1][i-1];
          v[jj-1][j-1]=x*c+z*s;
          v[jj-1][i-1]=z*c-x*s;
        }
        z=pythag(f,h);
        w[j-1]=z;
        if (z) {
          z=1.0/z;
          c=f*z;
          s=h*z;
        }
        f=c*g+s*y;
        x=c*y-s*g;
        for (jj=1;jj<=m;jj++) {
          y=a[jj-1][j-1];
          z=a[jj-1][i-1];
          a[jj-1][j-1]=y*c+z*s;
          a[jj-1][i-1]=z*c-y*s;
        }
      }
      rv1[l-1]=0.0;
      rv1[k-1]=f;
      w[k-1]=x;
    }
  }
  free_vector(rv1,1,n);
}