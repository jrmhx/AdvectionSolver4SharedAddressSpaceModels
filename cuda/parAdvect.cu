// CUDA parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2021
// v1.0 15 Apr 

// ./testAdvect [-h] [-s] [-g Gx[,Gy]] [-b Bx[,By]] [-o] [-w w] [-d d] M N [r]

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters

#define MAX_SHARED_MEMO 49152 // max number of bytes in shared memory on GPU per block 

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
// static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_, int verb) {
  M = M_;
  N = N_; 
  Gx = Gx_; 
  Gy = Gy_;  
  Bx = Bx_; 
  By = By_; 
  //verbosity = verb;
} //initParParams()


__host__ __device__
static void N2Coeff(double v, double *cm1, double *c0, double *cp1) {
  double v2 = v/2.0;
  *cm1 = v2*(v+1.0);
  *c0  = 1.0 - v*v;
  *cp1 = v2*(v-1.0);
}

__host__ __device__
void myUpdateAdvectField(int M, int N, double *u, int ldu, double *v, int ldv, double Ux, double Uy) {
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1);
  N2Coeff(Uy, &cjm1, &cj0, &cjp1);

  for (int i=0; i < M; i++)
  {
    for (int j=0; j < N; j++)
      {
        V(v,i,j) =
        cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
        ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
        cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
        //printf("update v(%d, %d) = %.2f\n", i, j, V(v,i,j));
      }
  }
} //updateAdvectField() 

__host__ __device__
void myCopyField(int M, int N, double *v, int ldv, double *u, int ldu) {
  for (int i=0; i < M; i++)
    for (int j=0; j < N; j++)
      V(u,i,j) = V(v,i,j);
}

__global__ void updateBoundaryNSKernel(int M, int N, double *u, int ldu) {
  // int j = blockIdx.y * blockDim.y + threadIdx.y;
  // int i = blockIdx.x * blockDim.x + threadIdx.x;
  // int j = blockIdx.y * blockDim.y + threadIdx.y;
  int xDim = blockDim.x * gridDim.x;
  int yDim = blockDim.y * gridDim.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int j = x*xDim + y; // map 2d thread pool in to 1d fashion
  
  while ( j < N + 2) {
    // printf("M, %d, N: %d, j: %d\n",M, N, j);
    // printf("(0, %d) = (%d, %d)\n", j, M, j);
    // printf("(%d, %d) = (1, %d)\n", M+1, j, j);
    // printf("\n");
    V(u, 0, j) = V(u, M, j);
    V(u, M+1, j) = V(u, 1, j);
    j += xDim * yDim;
  }
}

__global__ void updateBoundaryEWKernel(int M, int N, double *u, int ldu) {
  int xDim = blockDim.x * gridDim.x;
  int yDim = blockDim.y * gridDim.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int i = x*xDim + y;
  
  while (i < M + 2) {
    // printf("M, %d, N: %d, i: %d\n",M, N, i);
    // printf("(%d, 0) = (%d, %d)\n", i, i, N);
    // printf("(%d, %d) = (%d, 1)\n", i, N+1, i);
    // printf("\n");
    V(u, i, 0) = V(u, i, N);
    V(u, i, N+1) = V(u, i, 1);
    i += xDim * yDim;
  }
}

__global__ void updateAdvectFieldKernel(int M, int N, double *u, int ldu, double *v, int ldv, double Ux, double Uy) {
  // Compute unique thread indices within the grid
  int xDim = blockDim.x * gridDim.x;
  int yDim = blockDim.y * gridDim.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int M0 = (M / xDim) * x;
  int M_loc = (x < xDim - 1) ? (M / xDim) : (M - M0);

  int N0 = (N / yDim) * y;
  int N_loc = (y < yDim - 1) ? (N / yDim) : (N - N0);

  myUpdateAdvectField(M_loc, N_loc, &V(u, M0+1, N0+1), ldu, &V(v, M0+1, N0+1), ldv, Ux, Uy);
}

__global__ void copyFieldKernel(int M, int N, double *v, int ldu, double *u, int ldv) {
  // Compute unique thread indices within the grid
  int xDim = blockDim.x * gridDim.x;
  int yDim = blockDim.y * gridDim.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int M0 = (M / xDim) * x;
  int M_loc = (x < xDim - 1) ? (M / xDim) : (M - M0);

  int N0 = (N / yDim) * y;
  int N_loc = (y < yDim - 1) ? (N / yDim) : (N - N0);

  //printf("M0: %d, N0: %d, M_loc: %d, N_loc: %d\n", M0, N0, M_loc, N_loc);
  
  myCopyField(M_loc, N_loc, &V(v, M0+1, N0+1), ldu, &V(u, M0+1, N0+1), ldv);
}

__global__ void updateAdvectFieldOpt1(int M, int N, double *u, int ldu, double *v, int ldv, double cim1, double ci0, double cip1, double cjm1, double cj0, double cjp1) {
  extern __shared__ double s[];
  int lds = blockDim.x + 2;

  int si = threadIdx.x + 1;
  int sj = threadIdx.y + 1;

  int ui = blockIdx.x * blockDim.x + si;
  while (ui <= M){
    int uj = blockIdx.y * blockDim.y + sj;
    while (uj <= N){
      // printf("ui: %d, uj: %d\n", ui, uj);
      V(s, si, sj) = V(u, ui, uj);
      // __syncthreads();
      // update shared memo boundary
      if (si == 1) {
        V(s, si-1, sj) = V(u, ui-1, uj);
        if (sj == 1) {
          V(s, si-1, sj-1) = V(u, ui-1, uj-1);
        }
        if (sj == blockDim.y || uj == N) {
          V(s, si-1, sj+1) = V(u, ui-1, uj+1);
        }
      } 
      if (si == blockDim.x || ui == M) {
        V(s, si+1, sj) = V(u, ui+1, uj);
        if (sj == 1) {
          V(s, si+1, sj-1) = V(u, ui+1, uj-1);
        } 
        if (sj == blockDim.y || uj == N) {
          V(s, si+1, sj+1) = V(u, ui+1, uj+1);
        } 
      }
      //__syncthreads();
      if (sj == 1) {
        V(s, si, sj-1) = V(u, ui, uj-1);
      } 
      if (sj == blockDim.y || uj == N) {
        V(s,si, sj+1) = V(u, ui, uj+1);
      }
      __syncthreads();
      // update v
      
      V(v, ui ,uj) =
          cim1*(cjm1*V(s,si-1,sj-1) + cj0*V(s,si-1,sj) + cjp1*V(s,si-1,sj+1)) +
          ci0 *(cjm1*V(s,si  ,sj-1) + cj0*V(s,si,  sj) + cjp1*V(s,si,  sj+1)) +
          cip1*(cjm1*V(s,si+1,sj-1) + cj0*V(s,si+1,sj) + cjp1*V(s,si+1,sj+1));
      __syncthreads();
      uj += blockDim.y * gridDim.y;
    }
    ui += blockDim.x * gridDim.x;
  }
}

__global__ void updateAdvectFieldOpt2(int M, int N, double *u, int ldu, double *v, int ldv, double cim1, double ci0, double cip1, double cjm1, double cj0, double cjp1) {
  // Compute unique thread indices within the grid
  int xDim = blockDim.x * gridDim.x;
  int yDim = blockDim.y * gridDim.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int M0 = (M / xDim) * x;
  int M_loc = (x < xDim - 1) ? (M / xDim) : (M - M0);

  int N0 = (N / yDim) * y;
  int N_loc = (y < yDim - 1) ? (N / yDim) : (N - N0);
  //myUpdateAdvectField(M_loc, N_loc, &V(u, M0+1, N0+1), ldu, &V(v, M0+1, N0+1), ldv, Ux, Uy);
  u = &V(u, M0+1, N0+1);
  v = &V(v, M0+1, N0+1);
  for (int i=0; i < M_loc; i++)
  {
    for (int j=0; j < N_loc; j++)
      {
        V(v,i,j) =
        cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
        ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
        cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
        //printf("update v(%d, %d) = %.2f\n", i, j, V(v,i,j));
      }
  }
}

// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int reps, double *u, int ldu) {
  double Ux = Velx * dt / deltax;
  double Uy = Vely * dt / deltay;
  int ldv = N + 2;
  double *v;
  HANDLE_ERROR( cudaMalloc(&v, ldv*(M+2)*sizeof(double)) );

  dim3 block(Bx, By);
  dim3 grid(Gx, Gy);

  for (int r = 0; r < reps; r++) {
    //test block
    // updateBoundaryNS <<<1,1>>> (N, M, u, ldu);
    // updateBoundaryEW <<<1,1>>> (M, N, u, ldu);
    
    updateBoundaryNSKernel<<<grid, block>>>(M, N, u, ldu); 
    updateBoundaryEWKernel<<<grid, block>>>(M, N, u, ldu); 

    // test block
    // updateAdvectFieldK <<<1,1>>> (M, N, &V(u,1,1), ldu, &V(v,1,1), ldv, Ux, Uy);
    // copyFieldK <<<1,1>>> (M, N, &V(v,1,1), ldv, &V(u,1,1), ldu);

    updateAdvectFieldKernel<<<grid, block>>>(M, N, u, ldu, v, ldv, Ux, Uy); 
    copyFieldKernel <<<grid, block>>> (M, N, v, ldv, u, ldu); 
  } //for(r...)

  HANDLE_ERROR( cudaFree(v) );
} //cuda2DAdvect()



// ... optimized parallel variant
void cudaOptAdvect(int reps, double *u, int ldu, int w) {
  // if (M > Gx*Bx)
  //   if (M % (M / (Gx*Bx)) != 0 ){
  //     printf("Please set reasonable Gx and Bx, Gx*Bx is expected be a factor of M.\n");
  //     exit(0);
  //   }
  // if (N > Gy*By) 
  //   if (N % (N / (Gy*By)) != 0 ){
  //     printf("Please set reasonable Gy and By, Gy*By is expected to be a factor of N.\n");
  //     exit(0);
  //   }
  double Ux = Velx * dt / deltax;
  double Uy = Vely * dt / deltay;
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  N2Coeff(Ux, &cim1, &ci0, &cip1);
  N2Coeff(Uy, &cjm1, &cj0, &cjp1);

  int ldv = N + 2;
  double *v;
  HANDLE_ERROR( cudaMalloc(&v, ldv*(M+2)*sizeof(double)) );

  dim3 block(Bx, By);
  dim3 grid(Gx, Gy);

  for (int r = 0; r < reps; r++) {
    updateBoundaryNSKernel<<<grid, block>>>(M, N, u, ldu); 
    updateBoundaryEWKernel<<<grid, block>>>(M, N, u, ldu);
    bool isOpt1 = true;
    if (M > Gx*Bx){
      if (M % (M / (Gx*Bx)) != 0 ){
        isOpt1 = false;
      }
    }
    if (N > Gy*By){
      if (N % (N / (Gy*By)) != 0 ){
        isOpt1 = false;
      }
    } 
    if (isOpt1){
      size_t sharedMemSize = (Bx+2) * (By+2) * sizeof(double);
      //assert(sharedMemSize <= MAX_SHARED_MEMO);
      if (sharedMemSize > MAX_SHARED_MEMO) {
        printf("sharedMemo overflow with requested sharedMemSize: %lu, please try large grid and block size\n", sharedMemSize);
        exit(0);
      }
      updateAdvectFieldOpt1<<<grid, block, sharedMemSize>>>(M, N, u, ldu, v, ldv, cim1, ci0, cip1, cjm1, cj0, cjp1);
    } else {
      updateAdvectFieldOpt2<<<grid, block>>>(M, N, u, ldu, v, ldv, cim1, ci0, cip1, cjm1, cj0, cjp1);
    }
    cudaDeviceSynchronize();
    double *tmp = u;
    u = v;
    v = tmp;
    //HANDLE_ERROR( cudaMemcpy(u, v, ldv*(M+2)*sizeof(double), cudaMemcpyDeviceToDevice) );
  } //for(r...)
  if (reps % 2 == 1) {
    double *tmp = u;
    u = v;
    v = tmp;
    HANDLE_ERROR( cudaMemcpy(u, v, ldv*(M+2)*sizeof(double), cudaMemcpyDeviceToDevice) );
    //copyFieldKernel <<<grid, block>>> (M, N, v, ldv, u, ldu);
  }
  HANDLE_ERROR( cudaFree(v) );
} //cudaOptAdvect()
