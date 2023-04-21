// OpenMP parallel 2D advection solver module
// template written for COMP4300/8300 Assignment 2, 2021
// template v1.0 14 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "serAdvect.h" // advection parameters

static int M, N, P, Q; // local store of problem parameters
static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb) {
  M = M_, N = N_; P = P_, Q = Q_;
  verbosity = verb;
} //initParParams()


static void omp1dUpdateBoundary(double *u, int ldu) {
  int i, j;
  for (j = 1; j < N+1; j++) { //top and bottom halo
    V(u, 0, j)   = V(u, M, j);
    V(u, M+1, j) = V(u, 1, j);
  }
  for (i = 0; i < M+2; i++) { //left and right sides of halo 
    V(u, i, 0) = V(u, i, N);
    V(u, i, N+1) = V(u, i, 1);
  }
} 


static void omp1dUpdateAdvectField(double *u, int ldu, double *v, int ldv) {
  int i, j;
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1); N2Coeff(Uy, &cjm1, &cj0, &cjp1);
  for (i=0; i < M; i++)
    for (j=0; j < N; j++)
      V(v,i,j) =
        cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
        ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
        cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));

} //omp1dUpdateAdvectField()  


static void omp1dCopyField(double *v, int ldv, double *u, int ldu) {
  int i, j;
  for (i=0; i < M; i++)
    for (j=0; j < N; j++)
      V(u,i,j) = V(v,i,j);
} //omp1dCopyField()


// evolve advection over reps timesteps, with (u,ldu) containing the field
// using 1D parallelization
void omp1dAdvect(int reps, double *u, int ldu) {
  int r, ldv = N+2;
  double *v = calloc(ldv*(M+2), sizeof(double)); assert(v != NULL);
  for (r = 0; r < reps; r++) {    
    omp1dUpdateBoundary(u, ldu);
    omp1dUpdateAdvectField(&V(u,1,1), ldu, &V(v,1,1), ldv);
    omp1dCopyField(&V(v,1,1), ldv, &V(u,1,1), ldu);
  } //for (r...)
  free(v);
} //omp1dAdvect()


// ... using 2D parallelization
void omp2dAdvect(int reps, double *u, int ldu) {
  int i, j;
  int r, ldv = N+2;
  double *v = calloc(ldv*(M+2), sizeof(double)); assert(v != NULL);
  
  for (r = 0; r < reps; r++) {         
    for (j = 1; j < N+1; j++) { //top and bottom halo
      V(u, 0, j)   = V(u, M, j);
      V(u, M+1, j) = V(u, 1, j);
    }
    for (i = 0; i < M+2; i++) { //left and right sides of halo 
      V(u, i, 0) = V(u, i, N);
      V(u, i, N+1) = V(u, i, 1);
    }

    updateAdvectField(M, N, &V(u,1,1), ldu, &V(v,1,1), ldv);

    copyField(M, N, &V(v,1,1), ldv, &V(u,1,1), ldu);
  } //for (r...)
  free(v);
} //omp2dAdvect()


// ... extra optimization variant
void ompAdvectExtra(int reps, double *u, int ldu) {
  
} //ompAdvectExtra()
