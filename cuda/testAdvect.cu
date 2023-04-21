// CUDA 2D advection solver test program
// written by Peter Strazdins, Apr 21 for COMP4300/8300 Assignment 2
// v1.0 29 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> //getopt()
#include <assert.h>
#include <sys/time.h> //gettimeofday()
#include <string>   //std::string

#include "serAdvect.h"
#include "parAdvect.h"

#define USAGE   "testAdvect [-h] [-s] [-g Gx[,Gy]] [-b Bx[,By]] [-o] [-w w] [-v v] [-d d] M N [r]"
#define DEFAULTS "Gx=Gy=Bx=By=r=1 v=w=d=0"
#define OPTCHARS "hsg:b:ow:v:d:"

static int M, N;               // advection field size
static int Gx=1, Gy=1;         // grid dimensions
static int Bx=1, By=1;         // (thread) block dimensions
static int r = 1;              // number of timesteps for the simulation
static int optH = 0;           // set if -h specified
static int optS = 0;           // set if -s specified
static int optO = 0;           // set if -o specified
static int verbosity = 0;      // v, above
static int w = 0;              // optional extra tuning parameter
static int deviceNum = 0;      // d, above. id of GPU to be used

// print a usage message for this program and exit with a status of 1
void usage(std::string msg) {
  printf("testAdvect: %s\n", msg.c_str());
  printf("usage: %s\n\tdefault values: %s\n", USAGE, DEFAULTS);
  fflush(stdout);
  exit(1);
}

void getArgs(int argc, char *argv[]) {
  extern char *optarg; // points to option argument (for -p option)
  extern int optind;   // index of last option parsed by getopt()
  extern int opterr;
  int optchar;        // option character returned my getopt()
  int optD = 0;
  opterr = 0;          // suppress getopt() error message for invalid option
  while ((optchar = getopt(argc, argv, OPTCHARS)) != -1) {
    // extract next option from the command line     
    switch (optchar) {
    case 'h':
      optH = 1;
      break;
    case 's':
      optS = 1;
      break;
    case 'g':
      if (sscanf(optarg, "%d,%d", &Gx, &Gy) < 1) // invalid integer 
	usage("bad value for Gx");
      break;
    case 'b':
      if (sscanf(optarg, "%d,%d", &Bx, &By) < 1) // invalid integer 
	usage("bad value for Bx");
      break;
    case 'o':
      optO = 1;
      break;
    case 'w':
      if (sscanf(optarg, "%d", &w) != 1) // invalid integer 
	usage("bad value for w");
      break;
    case 'v':
      if (sscanf(optarg, "%d", &verbosity) != 1) // invalid integer 
	usage("bad value for v");
      break;
    case 'd':
      if (sscanf(optarg, "%d", &deviceNum) != 1) // invalid integer 
	usage("bad value for d");
      optD = 1;
      break;
    default:
      usage("unknown option");
      break;
    } //switch 
   } //while

  if (optind < argc) {
    if (sscanf(argv[optind], "%d", &M) != 1) 
      usage("bad value for M");
  } else
    usage("missing M");
  N = M;
  if (optind+1 < argc)
    if (sscanf(argv[optind+1], "%d", &N) != 1) 
      usage("bad value for N");
  if (optind+2 < argc)
    if (sscanf(argv[optind+2], "%d", &r) != 1) 
      usage("bad value for r");

  if (optH) //ignore -d
    deviceNum = 0;
  int maxDevices;
  HANDLE_ERROR( cudaGetDeviceCount(&maxDevices) );
  if (deviceNum < 0 || deviceNum >= maxDevices) {
    printf("warning: device id %d must be in range 0..%d. Using device 0.\n", 
	   deviceNum, maxDevices-1);
    deviceNum = 0;
  }
  if (optD)
    HANDLE_ERROR( cudaSetDevice(deviceNum) );
  HANDLE_ERROR( cudaGetDevice(&deviceNum) );

  cudaDeviceProp prop;
  HANDLE_ERROR( cudaGetDeviceProperties( &prop, deviceNum) );
  if (prop.maxThreadsPerBlock < Bx * By)
    printf("WARNING: Bx=%d By=%d too large for max threads per block = %d %s",
	   Bx, By, prop.maxThreadsPerBlock, "(EXPECT RUBBISH RESULTS)\n"); 
} //getArgs()


static void printAvgs(std::string name, double total, int nVals) {
  printf("%s %.3e\n", name.c_str(), total / nVals);
}

//return wall time in seconds
static double Wtime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return(1.0*tv.tv_sec + 1.0e-6*tv.tv_usec);
}

int main(int argc, char** argv) {
  double *u, *u_d = NULL; int ldu, uSize; //advection field
  double t, gflops, t_hd, t_dh; //times

  getArgs(argc, argv);

  printf("Advection of a %dx%d global field on %s %d" 
	 " for %d steps.\n", M, N, optH? "host": "GPU", deviceNum, r);
  if (optS)
    printf("\tusing serial computation\n");
  else if (optO)
    printf("\tusing optimizations (Gx,Gy=%d,%d Bx,By=%d,%d w=%d)\n", 
	   Gx, Gy, Bx, By, w);
  else if (!optH)
    printf("\tusing %dx%d blocks of %dx%d threads (2D decomposition)\n", 
	   Gx, Gy, Bx, By);  
  initAdvectParams(M, N);  
  initParParams(M, N, Gx, Gy, Bx, By, verbosity);

  ldu = N+2; uSize = (M+2)*ldu*sizeof(double); 
  u = (double *) calloc((M+2)*ldu, sizeof(double)); assert (u != NULL);

  initAdvectField(M, N, &V(u,1,1), ldu);
  if (verbosity > 1)
    printAdvectField("init u", M, N, &V(u,1,1), ldu);

  if (!optH) {
    HANDLE_ERROR( cudaMalloc(&u_d, uSize) );
    t_hd = Wtime();
    HANDLE_ERROR( cudaMemcpy(u_d, u, uSize, cudaMemcpyHostToDevice) );
    t_hd = Wtime() - t_hd;
  } 
    
  t = Wtime();
  if (optH)
    hostAdvectSerial(M, N, r, u, ldu);
  else if (optS)
    cudaAdvectSerial(M, N, r, u_d, ldu);
  else if (optO)    
    cudaOptAdvect(r, u_d, ldu, w); 
  else
    cuda2DAdvect(r, u_d, ldu);
  HANDLE_ERROR( cudaDeviceSynchronize() );
  t = Wtime() - t;

  gflops = 1.0e-09 * AdvFLOPsPerElt * M * N * r;
  printf("Advection time %.2es, GFLOPs rate=%.2e\n", t, gflops / t); 

  if (!optH) {
    t_dh = Wtime();
    HANDLE_ERROR( cudaMemcpy(u, u_d, uSize, cudaMemcpyDeviceToHost) );
    t_dh = Wtime() - t_dh;
    HANDLE_ERROR( cudaFree(u_d) );
    printf("Copy times: host-device %.2es, device-host %.2es\n", t_hd, t_dh);
  }
  
  if (verbosity > 1)
    printAdvectField("final u", M+2, N+2, u, ldu);
  printAvgs("Avg error of final field: ", 
	    errAdvectField(r, M, N, &V(u,1,1), ldu), M*N);
  printAvgs("Max error of final field: ", 
	    errMaxAdvectField(r, M, N, &V(u,1,1), ldu), 1);

  free(u);

  return 0;
} //main()

