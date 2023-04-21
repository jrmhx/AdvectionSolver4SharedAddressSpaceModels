// OpenMP 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2021 
// v1.0 01 Apr

// sets up parallel parameters above
void initParParams(int M, int N, int P, int Q, int verbosity);

// evolve advection over r timesteps, with (u,ldu) storing the local field
// using a 1D decomposition
void omp1dAdvect(int r, double *u, int ldu);

// 2D, wide parallel region variant
void omp2dAdvect(int r, double *u, int ldu);

// extra optimization variant
void ompAdvectExtra(int r, double *u, int ldu);
