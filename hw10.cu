/*


// kernel –find linearized threadId, and set A[ tid] = tid
__global__ void initArray( int *A) {
    // Write the array init function here
} 

int main(){
    // pointer for host and device memory
    int *h_a, *d_a;
    // define thread hierarchy
    int num_blocks = 8; 
    int num_th_per_blk = 8;
    
    // allocate host and device memory
    size_t mem_size;
    mem_size = num_blocks*num_th_per_blk*sizeof(int);
    h_a = (int*) malloc(mem_size);
    cudaMalloc((void**) &d_a, mem_size);
    
    //launch kernel
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    initArray<<<dimGrid, dimBlock>>>(d_a); 

    // retrieve results
    cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost);

    return 0;
} */

#include <stdio.h>
#include <assert.h>
#include <time.h>

__global__ void reverse_array(int *d, int count){
    extern __shared__ int s_data[];
 
    int inOffset  = blockDim.x * blockIdx.x;
    int in  = inOffset + threadIdx.x;
 
    // Store in reversed order into shared memory
    s_data[blockDim.x - 1 - threadIdx.x] = d_in[in];
 
    // Sync up the shared memory
    __syncthreads();
 
    // write from shared memory in forward order
    int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);
    int out = outOffset + threadIdx.x;
    d_out[out] = s_data[threadIdx.x];
    

    /*
    const int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid < count/2){
        const int new_tid = count - tid - 1;
        int prev = d[tid];
        d[tid] = d[new_tid];
        d[new_tid] = prev;
    }
    */
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
 
    // Copy host array to device array
    cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice);
 
    // launch kernel
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    reverse_array<<<dimGrid, dimBlock>>>(d_b, d_a, dimA);
 
    // block until the device has completed
    cudaThreadSynchronize();
 
    // device to host copy
    cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost);
 
    // verify the data returned to the host is correct
    for (int i = 0; i < dimA; i++){
        //assert(h_a[i] == h_b[dimA - 1 - i]);
    }

    printf("dimA = %d\n", dimA-1);
    printf("h_a[0] = %d and (dimA - 1 - i) = %d\n", h_a[0],  dimA - 1 - 0);
 
    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    // free host memory
    free(h_a);
    free(h_b);
 
    return 0;
}