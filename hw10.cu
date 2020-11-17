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
__global__ void reverseArray(int *A, int dimA){

    int tid, temp;
    tid = blockIdx.x* blockDim.x+ threadIdx.x; 

    if(tid < dimA/2){
        temp = A[tid];
        A[tid] = A[dimA-1-tid];
        A[dimA-1-tid] = temp;
    }
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
    cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice );

    // launch kernel
    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    reverseArray<<< dimGrid, dimBlock >>>(d_a, dimA);

    // block until the device has completed
    cudaThreadSynchronize();

    // device to host copy
    cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost );

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
    free(check);

    return 0;
} //qsub hw10.sh -q UI-GPU -I ngpus=1