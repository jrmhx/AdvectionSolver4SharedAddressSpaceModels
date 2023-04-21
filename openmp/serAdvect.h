// serial 2D advection solver module
// written by Peter Strazdins for COMP4300/8300 Assignment 1 
// v1.0 Feb 24

// number of FLOPs to update a single element in the advection function 
extern const int AdvFLOPsPerElt; 

// parameters needed for external advection solvers
extern const double Velx, Vely; //advection velocity
extern double dt;               //time for 1 step
extern double deltax, deltay;   //grid spacing       

// initializes the advection parameters for a global M x N field 
void initAdvectParams(int M, int N);

// access element (i,j) of array u with leading dimension ldu
#define V(u, i, j)       u[(i)*(ld##u) + (j)]
#define V_(u, ldu, i, j) u[(i)*(ldu)   + (j)]

// calculate 1D coefficients for the advection stencil
void N2Coeff(double v, double *cm1, double *c0, double *cp1);

//update 1 timestep for the local advection, without updating halos
//  the m x n row-major array (v,ldd) is updated from (u,ldu)
//  Assumes a halo of width 1 are around this array;
//  the corners of the halo are at u[-1,-1], u[-1,n], u[m,-1] and u[m,n]
void updateAdvectField(int m, int n, double *u, int ldu, double *v, int ldv);

// initialize (non-halo elements) of a m x n local advection field (u,ldu)
//    local element [0,0] is element [M0,N0] in the global field
void initAdvectField(int M0, int N0, int m, int n, double *u, int ldu);

// sum errors in an m x n local advection field (u,ldu) after r timesteps 
//    local element [0,0] is element [M0,N0] in the global field 
double errAdvectField(int r, int M0, int N0, int m, int n, double *u, int ldu);

//get abs max error in an m x n local advection field (u,ldu) after r timesteps
double errMaxAdvectField(int r, int M0, int N0, int m, int n, double *u, 
			 int ldu);

// print out the m x n local advection field (u,ldu) 
void printAdvectField(int rank, char *label, int m, int n, double *u, int ldu);

// copy m x n field (v, ldv) to ((u, ldu)
void copyField(int m, int n, double *v, int ldv, double *u, int ldu);
