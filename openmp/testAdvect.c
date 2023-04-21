// OpenMP 2D advection solver test program
// written by Peter Strazdins, Apr 21 for COMP4300/8300 Assignment 2
// v1.0 14 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> //getopt()
#include <assert.h>
#include <sys/time.h> //gettimeofday()
#include <omp.h>

#include "serAdvect.h"
#include "parAdvect.h"

#define USAGE   "OMP_NUM_THREADS=p testAdvect [-P P] [-x] [-v v] M N [r]"
#define DEFAULTS "P=p r=1 v=0"
#define OPTCHARS "P:xv:"

static int M, N;               // advection field size
static int P, Q;               // PxQ decomposition, Q = nprocs / P
static int r = 1;              // number of timesteps for the simulation
static int optP = 0;           // set if -P specified
static int optX = 0;           // set if -x specified
static int verbosity = 0;      // v, above
static int nprocs;             // p, above

// print a usage message for this program and exit with a status of 1
void usage(char *msg) {
  printf("testAdvect: %s\n", msg);
  printf("usage: %s\n\tdefault values: %s\n", USAGE, DEFAULTS);
  fflush(stdout);
  exit(1);
}

void getArgs(int argc, char *argv[]) {
  extern char *optarg; // points to option argument (for -p option)
  extern int optind;   // index of last option parsed by getopt()
  extern int opterr;
  char optchar;        // option character returned my getopt()
  opterr = 0;          // suppress getopt() error message for invalid option
  P = nprocs;
  while ((optchar = getopt(argc, argv, OPTCHARS)) != -1) {
    // extract next option from the command line     
    switch (optchar) {
    case 'P':
      if (sscanf(optarg, "%d", &P) != 1) // invalid integer 
	usage("bad value for P");
      optP = 1;
      break;
    case 'v':
      if (sscanf(optarg, "%d", &verbosity) != 1) // invalid integer 
	usage("bad value for v");
      break;
    case 'x':
      optX = 1;
      break;
    default:
      usage("unknown option");
      break;
    } //switch 
   } //while

  if (P == 0 || nprocs % P != 0)
    usage("number of threads must be a multiple of P");
  Q = nprocs / P;
  assert (Q > 0);

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
} //getArgs()


static void printAvgs(char *name, double total, int nVals) {
  printf("%s %.3e\n", name, total / nVals);
}

//return wall time in seconds
static double Wtime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return(1.0*tv.tv_sec + 1.0e-6*tv.tv_usec);
}

int main(int argc, char** argv) {
  double *u; int ldu; //advection field
  double t, gflops; //time

  nprocs = omp_get_max_threads();
  getArgs(argc, argv);

  printf("Advection of a %dx%d global field on %d threads" 
	 " for %d steps.\n", M, N, nprocs, r);
  if (optX)
    printf("\tusing extra optimization methods\n");
  if (optP)
    printf("\tusing a %dx%d decomposition\n", P, Q);
  
  initAdvectParams(M, N);  
  initParParams(M, N, P, Q, verbosity);

  ldu = N+2;
  u = calloc((M+2)*ldu, sizeof(double));
  initAdvectField(0, 0, M, N, &V(u,1,1), ldu);
  if (verbosity > 1)
    printAdvectField(0, "init u", M, N, &V(u,1,1), ldu);

  t = Wtime();
  if (optX)
    ompAdvectExtra(r, u, ldu);
  else if (optP)    
    omp2dAdvect(r, u, ldu); 
  else
    omp1dAdvect(r, u, ldu);
  t = Wtime() - t;

  gflops = 1.0e-09 * AdvFLOPsPerElt * M * N * r;
  printf("Advection time %.2es, GFLOPs rate=%.2e (per core %.2e)\n",
	 t, gflops / t,  gflops / t / (P*Q)); 

  if (verbosity > 1)
    printAdvectField(0, "final u", M+2, N+2, u, ldu);
  printAvgs("Avg error of final field: ", 
	    errAdvectField(r, 0, 0, M, N, &V(u,1,1), ldu), M*N);
  printAvgs("Max error of final field: ", 
	    errMaxAdvectField(r, 0, 0, M, N, &V(u,1,1), ldu), 1);

  free(u);
  return 0;
} //main()

