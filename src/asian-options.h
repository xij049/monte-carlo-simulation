/* Standard libraries */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* CUDA libraries */
#include <cuda.h>
#include <curand_kernel.h>

/* MCMC constants */
/* Number of blocks of simulation paths */
#ifndef NUM_BLOCKS
  #define NUM_BLOCKS 5
#endif
/* Number of paths per block */
#ifndef NUM_PATHS
  #define NUM_PATHS 128
#endif
/* Number of steps for each path */
#ifndef NUM_STEPS
  #define NUM_STEPS 3000
#endif
/* Number of warmup iterations before outputting the path */
#ifndef NUM_WARMUP_ITERATIONS
  #define NUM_WARMUP_ITERATIONS 1000
#endif

/* Options constants */
/* Starting price for the options */
#ifndef INITIAL_PRICE
  #define INITIAL_PRICE 50.0
#endif
/* Strike price */
#ifndef STRIKE_PRICE
  #define STRIKE_PRICE 150.0
#endif
/* Risk free rate */
#ifndef RISK_FREE_RATE
  #define RISK_FREE_RATE 0.03
#endif
/* Expected return */
#ifndef EXPECTED_RETURN
  #define EXPECTED_RETURN 0.04
#endif
/* Expected volume */
#ifndef EXPECTED_VOLUME
  #define EXPECTED_VOLUME 0.1
#endif
/* Expired time */
#ifndef EXPIRED_TIME
  #define EXPIRED_TIME 20.0
#endif
/* Time difference between steps */
#ifndef DT
  #define DT (EXPIRED_TIME / NUM_STEPS)
#endif

/* Error checker macro */
#define errorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
        exit(code);
      }
   }
}

/* Function headers */
/* Generates the options payoff by MCMC */
__global__ void MCMC_options(double *oldPaths, double *newPaths, double *oldZs, double *newZs, curandState* globalState);
/* Generates a path */
__device__ void generate_path(double *oldPath, double *newPath, double *oldZ, double *newZ, curandState *globalState);
/* Generates a new time step in a path */
__device__ double next_step(const double oldS, const double z);
/* Transition Probability Density Ratio */
__device__ double TPDR(const double s1, const double s2, const double rv1, const double rv2);
/* Expected Probability Density */
__device__ double EPD(const double s);
/* Generates a new random variable from N(0, 1) */
__device__ double random_normal(curandState* globalState);
/* Generates a new random variable from U(0, 1) */
__device__ double random_uniform(curandState* globalState);
