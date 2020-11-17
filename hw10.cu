#include <stdio.h>
#include <assert.h>
#include <time.h>

// kernel – reverse an array A
__global__ void reverseArray(int *A, int dim_a) {
    int tid, temp; 
    tid = blockIdx.x* blockDim.x+ threadIdx.x; 

    A[tid] = tid;

    /*
    if(tid < dim_a/2){
        temp = A[dim_a-tid-1];
        A[dim_a-tid-1] = A[tid];
        A[tid] = temp; 
    }*/
} 

int main() {
    //pointer for host memory and device memory 
    int *h_a, *d_a;
    
    // define thread hierarchy 
    int dim_a= 16*1024*1024;
    int num_blocks= 8; 
    int num_th_per_blk= dim_a/num_blocks; 
    
    // allocate host and device memory 
    size_t memSize; 
    memSize = num_blocks* num_th_per_blk* sizeof(int); 
    h_a = (int*) malloc(memSize);
    cudaMalloc( (void**) &d_a, memSize); 

    // Initialize input array on host
    for (int i = 0; i < dim_a; ++i) {
        //h_a[i] = i; //rand();
    }

    // store to device
    cudaMemcpy(d_a, h_a, memSize, cudaMemcpyDeviceToHost); 
    
    // launch kernel
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk); 
    reverseArray<<<dimGrid, dimBlock>>>(d_a, dim_a); 
    
    // retrieve results
    cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost); 

    // Initialize input array on host
    for (int i = 0; i < dim_a; ++i) {
        if(i == 15) break;
        printf("h_a[%d] = %d and (dim_a - 1 - %d) = %d\n", i, h_a[i], i,  dim_a - 1 - i);
    }

    // free device memory
    cudaFree(d_a);
    // free host memory
    free(h_a);
 
    return 0;
}