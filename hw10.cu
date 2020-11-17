#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <stdbool.h>

/*
    reverseArray
    @params
    @return void
*/
__global__ void reverseArray(int *d_out, int *d_in)
{
    extern __shared__ int s_data[];

    int inOffset  = blockDim.x * blockIdx.x;
    int in  = inOffset + threadIdx.x;

    // Load one element per thread from device memory and store it
    // in reversed order into temporary shared memory
    s_data[blockDim.x - 1 - threadIdx.x] = d_in[in];

    // Block until all threads in the block have written their data to shared mem
    __syncthreads();

    // write the data from shared memory in forward order,
    // but to the reversed block offset as before

    int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);

    int out = outOffset + threadIdx.x;
    d_out[out] = s_data[threadIdx.x];
}

/*
    main program
*/
int main(int argc, char** argv)
{
    // pointer for host memory and size
    int *h_a; 
    int dimA = 16*1024*1024; // 16 MB

    // array to compare results
    int *check;

    // pointer for device memory
    int *d_b, *d_a;

    // define grid and block size
    int numThreadsPerBlock = 8;

    // Compute number of blocks needed based on array size and desired block size
    int numBlocks = dimA / numThreadsPerBlock;

    // Part 1 of 2: Compute the number of bytes of shared memory needed
    // This is used in the kernel invocation below
    int sharedMemSize = numThreadsPerBlock * sizeof(int);

    // allocate host and device memory
    size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
    h_a = (int *) malloc(memSize);
    check = (int *) malloc(memSize);
    cudaMalloc((void **) &d_a, memSize);
    cudaMalloc((void **) &d_b, memSize);


    // Initialize input array on host
    int val;
    srand(time(0));
    for (int i = 0; i < dimA; ++i)
    {
        val = rand();
        h_a[i] = val;
        check[i] = val;
    }

    // Copy host array to device array
    cudaMemcpy( d_a, h_a, memSize, cudaMemcpyHostToDevice );

    // launch kernel
    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    reverseArray<<< dimGrid, dimBlock, sharedMemSize >>>(d_b, d_a);

    // block until the device has completed
    cudaThreadSynchronize();

    // device to host copy
    cudaMemcpy( h_a, d_b, memSize, cudaMemcpyDeviceToHost );

    printf("Verifying program correctness.... ");
    // verify the data returned to the host is correct
    for (int i = 0; i < dimA; i++)
    {
        assert(h_a[i] == check[dimA - 1 - i]);
    }
    printf("Everthing checks out!\n");

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);

    // free host memory
    free(h_a);
    free(test_arr);

    return 0;
}