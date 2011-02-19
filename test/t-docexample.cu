/*
* This program uses the host CURAND API to generate 100
* pseudorandom floats.
*/

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <curand.h>

class RNG {};

class QRNG : public RNG {};

class PRNG : public PRNG {};

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {	\
	    printf("Error at %s:%d\n",__FILE__,__LINE__);	\
	    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) {	\
	    printf("Error at %s:%d\n",__FILE__,__LINE__);	\
	    return EXIT_FAILURE;}} while(0)

int main(int argc, char *argv[])
{
    size_t n = 5;
    size_t i;
    curandGenerator_t gen;
    float *devData, *hostData;

    /* Allocate n floats on host */
    //hostData = (float *)calloc(n, sizeof(float));

    hostData = new float[n];

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc((void **)&devData, n * sizeof(float)));

    /* Create pseudo-random number generator */
    // CURAND_CALL(curandCreateGenerator(&gen,
    // 				      CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandCreateGenerator(&gen,
    				      CURAND_RNG_QUASI_SOBOL32));

    /* Set seed */
    // CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, devData, n));

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float),
			 cudaMemcpyDeviceToHost));

    /* Show result */
    for(i = 0; i < n; i++)
	{
	    std::cout << hostData[i] << " ";
	}
    std::cout << std::endl;

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    delete hostData;

    return EXIT_SUCCESS;
}
