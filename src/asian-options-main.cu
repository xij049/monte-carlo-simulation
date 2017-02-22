/* Headers and definitions */
#include "asian-options.h"

/* RNG initializer */
__global__ void setup_rng(curandState *state, unsigned long seed) {
    int idx = threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

/* Main procedure */
int main(const int argc, const char** argv) {
  /* Final paths */
  double finalPaths[NUM_PATHS * NUM_STEPS];
  const size_t pathsSize = NUM_PATHS * NUM_STEPS;

  /* RNG states */
  curandState *devStates;
  errorCheck(cudaMalloc(&devStates, NUM_PATHS * sizeof(curandState)));
  setup_rng<<< 1, NUM_PATHS >>>(devStates, time(NULL));
  errorCheck(cudaPeekAtLastError());

  /* Device memory:
   *   Old paths
   *   New paths
   *   Old Z's
   *   New Z's
   */
  double *paths;
  errorCheck(cudaMalloc(&paths, 4 * pathsSize * sizeof(double)));

  /* Generate and print the paths */
  for (unsigned long i = 0; i < NUM_BLOCKS; ++i) {
    MCMC_options<<< 1, NUM_PATHS >>>(paths, paths + pathsSize, paths + 2 * pathsSize, paths + 3 * pathsSize, devStates);
    errorCheck(cudaPeekAtLastError());
    errorCheck(cudaDeviceSynchronize());

    /* Copy the paths from CUDA to host */
    errorCheck(cudaMemcpy(finalPaths, &paths[pathsSize], pathsSize * sizeof(double), cudaMemcpyDeviceToHost));
    errorCheck(cudaDeviceSynchronize());

    /* Print the paths */
    for (unsigned long j = 0; j < NUM_PATHS; ++j) {
      for (unsigned long k = 0; k < (NUM_STEPS - 1); ++k) {
        printf("%.4lf,", finalPaths[NUM_STEPS * j + k]);
      }
      printf("%.4lf\n", finalPaths[NUM_STEPS * (j + 1) - 1]);
    }
  }

  return 0;
}
