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

__global__ void gainCalculator(int *d_candidates, size_t num_candidates, int* d_medoids, size_t num_medoids, float* d_distanceMatrix, float* d_res, int N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_candidates){
        d_res[i] = 0;
        for(int j = 0; j < num_candidates; j++){
            if(d_candidates[j] == d_candidates[i]) continue;
            float dissimilarity = std::numeric_limits<float>::max();
            for(int k = 0; k < num_medoids; k++){
                float dist = d_distanceMatrix[d_candidates[j]*N + d_medoids[k]];
                dissimilarity = std::min(dissimilarity, dist);
            }
            d_res[i] += std::max(dissimilarity - d_distanceMatrix[d_candidates[j]*N + d_candidates[i]], (float)0);
        }
    }
}

int* set_to_array(std::unordered_set<int>& s){
    int *v = new int[s.size()];
    int idx = 0;
    for(auto item: s){
        v[idx++] = item;
    }
    return v;
}

float* gainCalculatorWrapper(int* h_candidates, size_t num_candidates, int* h_medoids, size_t num_medoids, float* d_distanceMatrix, int N){
    // Allocating all memory
    int *d_candidates, *d_mediods; 
    float *d_res, *h_res;
    cudaMalloc((void**)&d_candidates, num_candidates*sizeof(int));
    cudaMalloc((void**)&d_mediods, num_medoids*sizeof(int));
    cudaMalloc((void**)&d_res, num_candidates*sizeof(float));
    h_res = new float[num_candidates];

    // Copying from host to device
    cudaMemcpy(d_candidates, h_candidates, num_candidates*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mediods, h_medoids, num_medoids*sizeof(int), cudaMemcpyHostToDevice);


    // setting up the kernel
    int blockSize = 256;
    int numBlocks = (num_candidates + blockSize - 1)/blockSize;
    gainCalculator<<<numBlocks, blockSize>>>(d_candidates, num_candidates, d_mediods, num_medoids, d_distanceMatrix, d_res, N);
    
    //Copying the result back
    cudaMemcpy(h_res, d_res, num_candidates*sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory 
    cudaFree(d_candidates);
    cudaFree(d_mediods);
    cudaFree(d_res);

    cudaDeviceSynchronize();

    return h_res;
}