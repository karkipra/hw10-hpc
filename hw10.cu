#include <stdio.h>
#include <assert.h>
#include <time.h>

__global__ void reverse_array(int *d, int count){
    int tid;
    tid = blockIdx.x* blockDim.x+ threadIdx.x; 
    if(tid < count/2){
        int prev = d[tid];
        d[tid] = d[count - tid - 1];
        d[count - tid - 1] = prev;
    }
}

int main( int argc, char** argv) {
    // pointer for host memory and size
    int *h_a;
    int dimA = 16 * 1024 * 1024; // 16MB
    // pointer for comparing results
    int *h_b;
 
    // pointer for device memory
    int *d_a, *d_b;
 
    // define grid and block size
    int num_th_per_blk = 16;
    // Compute number of blocks needed based on array size and desired block size
    int num_blocks = dimA / num_th_per_blk;  
 
    // allocate host and device memory
    size_t mem_size;
    mem_size = num_blocks * num_th_per_blk * sizeof(int);
    h_a = (int *) malloc(mem_size);
    h_b = (int *) malloc(mem_size);
    cudaMalloc((void **) &d_a, mem_size);
    cudaMalloc((void **) &d_b, mem_size);

    // seed the rand()
    srand(time(NULL));
 
    // Initialize input array on host
    for (int i = 0; i < dimA; ++i) {
        h_a[i] = i; //rand();
        h_b[i] = h_a[i];
    }

    printf("h_a[0] = %d and h_b[0] = %d\n", h_a[0], h_b[0]);
 
    // launch kernel
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    reverse_array<<<dimGrid, dimBlock>>>(d_a, dimA);
 
    // device to host copy
    cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost);

    printf("dimA = %d\n", dimA-1);
 
    // verify the data returned to the host is correct
    for (int i = 0; i < dimA; i++){
        if(i == 15) break;
        printf("h_a[%d] = %d and (dimA - 1 - %d) = %d\n", i, h_a[i], i,  dimA - 1 - i);
    }
    
    //printf("h_a[0] = %d and (dimA - 1 - i) = %d\n", h_a[0],  dimA - 1 - 0);
 
    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    // free host memory
    free(h_a);
    free(h_b);
 
    return 0;
}