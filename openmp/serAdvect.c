// serial 2D advection solver module
// written by Peter Strazdins for COMP4300/8300 Assignment 1 
// v1.0 25 Feb

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h> // sin(), fabs()

#include "serAdvect.h"


// advection parameters
static const double CFL = 0.25;   // CFL condition number
const double Velx = 1.0, Vely = 1.0; //advection velocity
double dt;                 //time for 1 step
double deltax, deltay;     //grid spacing//

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

void initAdvectField(int M0, int N0, int m, int n, double *u, int ldu) {
  int i, j; 
  for (i=0; i < m; i++) {
    double y = deltay * (i + M0);
    for (j=0; j < n; j++) {
      double x = deltax * (j + N0);
      V(u, i, j) = initCond(x, y, 0.0);
    }
  }
} //initAdvectField()


double errAdvectField(int r, int M0, int N0, int m, int n, double *u, int ldu){
  int i, j;
  double err = 0.0;
  double t = r * dt;
  for (i=0; i < m; i++) {
    double y = deltay * (i + M0);
    for (j=0; j < n; j++) {
      double x = deltax * (j + N0);
      err += fabs(V(u, i, j) - initCond(x, y, t));
    }
  }
  return (err);
} //errAdvectField()


double errMaxAdvectField(int r, int M0, int N0, int m, int n, 
			 double *u, int ldu) {
  int i, j;
  double err = 0.0;
  double t = r * dt;
  for (j=0; j < n; j++) {
    double x = deltax * (j + N0);
    for (i=0; i < m; i++) {
      double y = deltay * (i + M0);
      double e = fabs(V(u, i, j) - initCond(x, y, t));
      if (e > err)
	err = e;
    }
  }
  return (err);
}


const int AdvFLOPsPerElt = 20; //count 'em

void N2Coeff(double v, double *cm1, double *c0, double *cp1) {
  double v2 = v/2.0;
  *cm1 = v2*(v+1.0);
  *c0  = 1.0 - v*v;
  *cp1 = v2*(v-1.0);
}

// uses the Lax-Wendroff method
void updateAdvectField(int m, int n, double *u, int ldu, double *v, int ldv) {
  int i, j;
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  double cim1, ci0, cip1;
  double cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1);
  N2Coeff(Uy, &cjm1, &cj0, &cjp1);

  for (i=0; i < m; i++) 
    for (j=0; j < n; j++)
      V(v,i,j) =
        cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
        ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
	cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));

} //updateAdvectField()


void printAdvectField(int rank, char *label, int m, int n, double *u, int ldu){
  int i, j;
  printf("%d: %s\n", rank, label);
  for (i=0; i < m; i++) {
    printf("%d: ", rank);  
    for (j=0; j < n; j++) 
      printf(" %+0.2f", V(u, i, j));
    printf("\n");
  }
}


void copyField(int m, int n, double *v, int ldv, double *u, int ldu) {
  int i, j;
  for (i=0; i < m; i++)
    for (j=0; j < n; j++)
      V(u,i,j) = V(v,i,j);
}
