#include <stdio.h>
#include <assert.h>
#include <time.h>
/*
__global__ void reverse_array(int *d_a, int dim_a){
    int tid;
    tid = blockIdx.x* blockDim.x+ threadIdx.x; 
    if(tid < count/2){
        int rev = dim_a - tid - 1;
        int prev = d_a[tid];
        d_a[tid] = d_a[dim_a - tid - 1];
        d_a[dim_a - tid - 1] = prev;
    }
}

int main() {
    // pointer for host memory and size
    int *h_a;
    int dim_a = 16 * 1024 * 1024; // 16MB
    // pointer for comparing results
    int *h_b;
 
    // pointer for device memory
    int *d_a, *d_b;
 
    // define grid and block size
    int num_th_per_blk = 16*1024;
    // Compute number of blocks needed based on array size and desired block size
    int num_blocks = dim_a / num_th_per_blk;  
 
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
    for (int i = 0; i < dim_a; ++i) {
        h_a[i] = i; //rand();
        h_b[i] = h_a[i];
    }

    printf("h_a[0] = %d and h_b[0] = %d\n", h_a[0], h_b[0]);

    // host to device copy
    cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice);
 
    // launch kernel
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    reverse_array<<<dimGrid, dimBlock>>>(d_a, dim_a);
 
    // device to host copy
    cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost);

    printf("dim_a = %d\n", dim_a-1);
 
    // verify the data returned to the host is correct
    for (int i = 0; i < dim_a; i++){
        if(i == 15) break;
        printf("h_a[%d] = %d and (dim_a - 1 - %d) = %d\n", i, h_a[i], i,  dim_a - 1 - i);
    }
    
    //printf("h_a[0] = %d and (dim_a - 1 - i) = %d\n", h_a[0],  dim_a - 1 - 0);
 
    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    // free host memory
    free(h_a);
    free(h_b);
 
    return 0;
}*/

// kernel – reverse an array A
__global__ void reverseArray(int *A, int dim_a) {
    int tid, temp, N; 
    tid = blockIdx.x* blockDim.x+ threadIdx.x; 
    N = blockDim.x * gridDim.x;

    A[tid] = tid;

    /*
    if(tid < dim_a/2){
        temp = A[N-tid-1];
        A[N-tid-1] = A[tid];
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