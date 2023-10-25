#pragma once

__global__ void distamceSumAllPoints(int N, float* d_distanceMatrix, float* d_res){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < N){
        d_res[i] = 0;
        for(int j = 0; j < N; j++) d_res[i] += d_distanceMatrix[i*N + j];
    }
}

float* distanceSumAllPointsWrapper(int N, float* d_distanceMatrix){
    // allocating memory
    float *d_res, *h_res;
    cudaMalloc((void**)&d_res, N*N*sizeof(float));
    h_res = new float[N];

    //setting up the kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1)/blockSize;
    distamceSumAllPoints<<<numBlocks, blockSize>>>(N, d_distanceMatrix, d_res);
    

    //copying back to the device
    cudaMemcpy(h_res, d_res, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_res);
    cudaDeviceSynchronize();

    return h_res;
}   