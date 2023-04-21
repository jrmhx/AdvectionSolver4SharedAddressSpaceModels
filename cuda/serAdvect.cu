// serial 2D advection solver module
// written by Peter Strazdins, Apr 21, for COMP4300/8300 Assignment 1
// v1.0 29 Apr

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h> // sin(), fabs()

#include "serAdvect.h"

void cudaHandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

// advection parameters
static const double CFL = 0.25;      //CFL condition number
const double Velx = 1.0, Vely = 1.0; //advection velocity
double dt;                           //time for 1 step
double deltax, deltay;               //grid spacing

void initAdvectParams(int M, int N) {
  assert (M > 0 && N > 0); // advection not defined for empty grids
  deltax = 1.0 / N;
  deltay = 1.0 / M;
  dt = CFL * (deltax < deltay? deltax: deltay);
} 

static double initCond(double x, double y, double t) {
  x = x - Velx*t;
  y = y - Vely*t;
  return (sin(4.0*M_PI*x) * sin(2.0*M_PI*y)) ;
}

void initAdvectField(int M, int N, double *u, int ldu) {
  int i, j;
  for (i=0; i < M; i++) {
    double y = deltay * i;
    for (j=0; j < N; j++) {
      double x = deltax * j;
      V(u, i, j) = initCond(x, y, 0.0);
    }
  }
} //initAdvectField()

double errAdvectField(int r, int M, int N, double *u, int ldu){
  int i, j;
  double err = 0.0;
  double t = r * dt;
  for (i=0; i < M; i++) {
    double y = deltay * i;
    for (j=0; j < N; j++) {
      double x = deltax * j;
      err += fabs(V(u, i, j) - initCond(x, y, t));
    }
  }
  return (err);
} //errAdvectField()

double errMaxAdvectField(int r, int M, int N, double *u, int ldu) {
  int i, j;
  double err = 0.0;
  double t = r * dt;
  for (i=0; i < M; i++) {
    double y = deltay * i;
    for (j=0; j < N; j++) {
      double x = deltax * j;
      double e = fabs(V(u, i, j) - initCond(x, y, t));
      if (e > err)
        err = e;
    }
  }
  return (err);
} //errMaxAdvectField()

void printAdvectField(std::string label, int M, int N, double *u, int ldu) {
  int i, j;
  printf("%s\n", label.c_str());
  for (i=0; i < M; i++) {
    for (j=0; j < N; j++) 
      printf(" %+0.2f", V(u, i, j));
    printf("\n");
  }
}

const int AdvFLOPsPerElt = 20; //count 'em

__host__ __device__
static void N2Coeff(double v, double *cm1, double *c0, double *cp1) {
  double v2 = v/2.0;
  *cm1 = v2*(v+1.0);
  *c0  = 1.0 - v*v;
  *cp1 = v2*(v-1.0);
}

__host__ __device__
void updateAdvectField(int M, int N, double *u, int ldu, double *v, int ldv,
		       double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1);
  N2Coeff(Uy, &cjm1, &cj0, &cjp1);

  for (int i=0; i < M; i++)
    for (int j=0; j < N; j++)
      V(v,i,j) =
        cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
        ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
        cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));

} //updateAdvectField() 

__host__ __device__
void copyField(int M, int N, double *v, int ldv, double *u, int ldu) {
  for (int i=0; i < M; i++)
    for (int j=0; j < N; j++)
      V(u,i,j) = V(v,i,j);
}

static void updateBoundary(int M, int N, double *u, int ldu) {
  for (int j = 1; j < N+1; j++) { //top and bottom halo
    V(u, 0, j)   = V(u, M, j);
    V(u, M+1, j) = V(u, 1, j);
  }
  for (int i = 0; i < M+2; i++) { //left and right sides of halo
    V(u, i, 0) = V(u, i, N);
    V(u, i, N+1) = V(u, i, 1);
  }
}


void hostAdvectSerial(int M, int N, int reps, double *u, int ldu) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N+2;
  double *v = (double *) calloc(ldv*(M+2), sizeof(double)); assert(v != NULL);
  for (int r = 0; r < reps; r++) {
    updateBoundary(M, N, u, ldu);
    updateAdvectField(M, N, &V(u,1,1), ldu, &V(v,1,1), ldv, Ux, Uy);
    copyField(M, N, &V(v,1,1), ldv, &V(u,1,1), ldu);
  } //for(r...)
  free(v);
} //hostAdvectSerial()


/********************** serial GPU area **********************************/

__global__
void updateBoundaryNS(int N, int M, double *u, int ldu) {
  for (int j=1; j < N+1; j++) { //top and bottom halo
    V(u, 0, j)   = V(u, M, j);
    V(u, M+1, j) = V(u, 1, j);
  }
}

__global__
void updateBoundaryEW(int M, int N, double *u, int ldu) {
  for (int i=0; i < M+2; i++) { //left and right sides of halo
    V(u, i, 0) = V(u, i, N);
    V(u, i, N+1) = V(u, i, 1);
  }
}

__global__
void updateAdvectFieldK(int M, int N, double *u, int ldu, double *v, int ldv,
			double Ux, double Uy) {
  updateAdvectField(M, N, u, ldu, v, ldv, Ux, Uy);
}

__global__
void copyFieldK(int M, int N, double *u, int ldu, double *v, int ldv) {
  copyField(M, N, u, ldu, v, ldv);
}

// evolve advection over reps timesteps, with (u,ldu) containing the field
void cudaAdvectSerial(int M, int N, int reps, double *u, int ldu) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N+2; double *v;
  HANDLE_ERROR( cudaMalloc(&v, ldv*(M+2)*sizeof(double)) );
  for (int r = 0; r < reps; r++) {
    updateBoundaryNS <<<1,1>>> (N, M, u, ldu);
    updateBoundaryEW <<<1,1>>> (M, N, u, ldu);
    updateAdvectFieldK <<<1,1>>> (M, N, &V(u,1,1), ldu, &V(v,1,1), ldv,
				  Ux, Uy);
    copyFieldK <<<1,1>>> (M, N, &V(v,1,1), ldv, &V(u,1,1), ldu);
  } //for(r...)
  HANDLE_ERROR( cudaFree(v) );
} //cudaAdvectSerial()

