// CUDA 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2021
// v1.0 15 Apr

//sets up parallel parameters above
void initParParams(int M, int N, int Gx, int Gy, int Bx, int By, int verb);

// evolve advection on GPU over r timesteps, with (u,ldu) storing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int r, double *u, int ldu);

// optimized parallel variant
void cudaOptAdvect(int r, double *u, int ldu, int w);
