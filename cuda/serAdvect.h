// serial 2D advection solver module
// written by Peter Strazdins, Apr 21, for COMP4300/8300 Assignment 2
// v1.0 14 Apr
#include <string>   //std::string

#define HANDLE_ERROR( err ) (cudaHandleError( err, __FILE__, __LINE__ ))
void cudaHandleError(cudaError_t err, const char *file, int line);

// number of FLOPs to update a single element in the advection function 
extern const int AdvFLOPsPerElt; 

// parameters needed for advection solvers
extern const double Velx, Vely; //advection velocity                
extern double dt;               //time for 1 step
extern double deltax, deltay;   //grid spacing

// initializes the advection parameters for a global M x N field 
void initAdvectParams(int M, int N);

// access element (i,j) of array u with leading dimension ldu
#define V(u, i, j)       u[(i)*(ld##u) + (j)]
#define V_(u, ldu, i, j) u[(i)*(ldu)   + (j)]

// initialize (non-halo elements) of an M x N advection field (u, ldu)
void initAdvectField(int M, int N, double *u, int ldu);

// sum errors in an M x N advection field (u, ldu) after r timesteps 
double errAdvectField(int r, int M, int N, double *u, int ldu);

//get abs max error in an M x N advection field (u,ldu) after r timesteps
double errMaxAdvectField(int r, int M, int N, double *u, int ldu);

// print out the m x n advection field (u, ldu) 
void printAdvectField(std::string label, int m, int n, double *u, int ldu);

#if 0 //for some reason, nvcc would not export this function properly
// calculate 1D coefficients for the advection stencil
__host__ __device__
void N2Coeff(double v, double *cm1, double *c0, double *cp1);
#endif

//update 1 timestep for the local advection, without updating halos
//  the M x N row-major array (v,ldv) is updated from (u,ldu)
//  Assumes a halo of width 1 are around this array;
//  the corners of the halo are at u[-1,-1], u[-1,n], u[m,-1] and u[m,n]
__host__ __device__
void updateAdvectField(int M, int N, double *u, int ldu, double *v, int ldv,
		       double Ux, double Uy);

// copy M x N field (v, ldv) to (u, ldu)
__host__ __device__
void copyField(int M, int N, double *v, int ldv, double *u, int ldu);


// evolve advection on host over r timesteps, with (u,ldu) storing the field
void hostAdvectSerial(int M, int N, int r, double *u, int ldu);

// evolve advection on GPU over r timesteps, with (u,ldu) storing the field
void cudaAdvectSerial(int M, int N, int r, double *u, int ldu);

// kernels that it uses
__global__
void updateBoundaryEW(int M, int N, double *u, int ldu);
__global__
void updateBoundaryNS(int N, int M, double *u, int ldu);
__global__
void updateAdvectFieldK(int M, int N, double *u, int ldu, double *v, int ldv,
			double Ux, double Uy);
__global__
void copyFieldK(int M, int N, double *v, int ldv, double *u, int ldu);

