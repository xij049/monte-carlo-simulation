/* Headers and definitions */
#include "asian-options.h"

/* Function implementations */
__global__ void MCMC_options(double *oldPaths, double *newPaths, double *oldZs, double *newZs, curandState* globalState) {
  double *oldPath = &oldPaths[threadIdx.x * NUM_STEPS];
  double *newPath = &newPaths[threadIdx.x * NUM_STEPS];
  double *oldZ = &oldZs[threadIdx.x * NUM_STEPS];
  double *newZ = &newZs[threadIdx.x * NUM_STEPS];
  generate_path(oldPath, newPath, oldZ, newZ, globalState);
}

__device__ void generate_path(double *oldPath, double *newPath, double *oldZ, double *newZ, curandState *globalState) {
  /* First path */
  oldPath[0] = INITIAL_PRICE;
  for (unsigned long j = 1; j < NUM_STEPS; ++j) {
    oldZ[j] = random_normal(globalState);
    oldPath[j] = next_step(oldPath[j-1], oldZ[j]);
  }

  /* Rest of the paths */
  for (unsigned long i = 1; i < NUM_WARMUP_ITERATIONS; ++i) {
    /* Starting price */
    newPath[0] = INITIAL_PRICE;

    /* Fill the path using previous path */
    for (unsigned j = 1; j < NUM_STEPS; ++j) {
      newZ[j] = random_normal(globalState);

      const double oldS = oldPath[j];
      const double newS = next_step(oldS, newZ[j]);

      // const double r = EPD(newS) / EPD(oldS) * TPDR(newS, oldS, newZ[j], oldZ[j]);
      const double r = EPD(newS) / EPD(oldS);
      newPath[j] = (random_uniform(globalState) < r) ? newS : oldS;
    }

    /* Save last path */
    for (unsigned long j = 1; j < NUM_STEPS; ++j) {
      oldPath[j] = newPath[j];
      oldZ[j] = newZ[j];
    }
  }
}

__device__ double next_step(const double oldS, const double z) {
  return oldS * exp((EXPECTED_RETURN - 0.5 * EXPECTED_VOLUME) * DT
                    + z * sqrt(EXPECTED_VOLUME * DT));
}

__device__ double TPDR(const double s1, const double s2, const double rv1, const double rv2) {
  const double pSum = (EXPECTED_RETURN - 0.5 * EXPECTED_VOLUME) * DT;
  const double pFactor = sqrt(EXPECTED_VOLUME * DT);
  const double dP1 = pSum + pFactor * rv1;
  const double dP2 = pSum + pFactor * rv2;
  return exp(-abs(log(s2) - log(s1) - dP1) / DT) /
         exp(-abs(log(s1) - log(s2) - dP2) / DT);
}

__device__ double EPD(const double s) {
  const double sq = log(s) - log(INITIAL_PRICE);
  return exp(- sq * sq / (2.0 * EXPECTED_VOLUME));
}

__device__ double random_normal(curandState* globalState) {
  int idx = threadIdx.x;
  curandState localState = globalState[idx];

  double z = curand_normal(&localState);
  globalState[idx] = localState;

  return z;
}

__device__ double random_uniform(curandState* globalState) {
  int idx = threadIdx.x;
  curandState localState = globalState[idx];

  double z = curand_uniform(&localState);
  globalState[idx] = localState;

  return z;
}
