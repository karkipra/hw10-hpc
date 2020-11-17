#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <stdbool.h>

/*
    References 
        - https://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/208801731?pgno=2 
*/

/*
    reverseArray - reverses an array in kernel
    @params int*A, int dim_a
    @return void
*/
__global__ void reverseArray(int *A, int dim_a){

    int tid, temp;
    tid = blockIdx.x* blockDim.x+ threadIdx.x; 

    if(tid < dim_a/2){
        temp = A[tid];
        A[tid] = A[dim_a-1-tid];
        A[dim_a-1-tid] = temp;
    }
}

/*
    main program
*/
int main(int argc, char** argv)
{
    // pointer for host memory and size
    int *h_a; 
    int dim_a = 16*1024*1024; // 16 MB

    // array to compare results
    int *check;

    // pointer for device memory
    int *d_a;

    // define grid and block size
    int num_th_per_blk = 8;
    int num_blocks = dim_a / num_th_per_blk;

    // allocate host and device memory
    size_t memSize = num_blocks * num_th_per_blk * sizeof(int);
    h_a = (int *) malloc(memSize);
    check = (int *) malloc(memSize);
    cudaMalloc((void **) &d_a, memSize);

    // Initialize input array on host
    int val;
    srand(time(NULL));
    for (int i = 0; i < dim_a; i++){
        val = rand();
        h_a[i] = val;
        check[i] = val;
    }

    // Copy host array to device array
    cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice );

    // launch kernel
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    reverseArray<<< dimGrid, dimBlock >>>(d_a, dim_a);

    // device to host copy
    cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost );

    printf("Verifying program correctness.... ");
    // verify the data returned to the host is correct
    for (int i = 0; i < dim_a; i++){
        assert(h_a[i] == check[dim_a - 1 - i]);
    }
    printf("Everthing checks out!\n");

    // free device memory
    cudaFree(d_a);
    // free host memory
    free(h_a);
    free(check);

    return 0;
} //qsub hw10.sh -q UI-GPU -I ngpus=1