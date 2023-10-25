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


// ! frees the input array !
__device__ float* findSmallestandSecondSmallest(float* v, size_t n){
    float *d_res;
    cudaMalloc((void**)&d_res, 2*sizeof(float));

    int idx = -1;
    float smallest = std::numeric_limits<float>::max();
    for(int i = 0; i < n; i++){
        if(v[i] < smallest){
            smallest = v[i];
            idx = i;
        }
    }

    float secondSmallest = std::numeric_limits<float>::max();
    for(int i = 0; i < n; i++){
        if(i == idx) continue;
        if(v[i] < secondSmallest) secondSmallest = v[i];
    }

    d_res[0] = smallest;
    d_res[1] = secondSmallest;
    cudaFree(v);

    return d_res;
}

__device__ float* findDE(int i, int* medoids, size_t num_medoids, float *d_distaneMatrix, int N){
    float *d_clusterDistances;
    cudaMalloc((void**)&d_clusterDistances, num_medoids*sizeof(float));

    for(int j = 0; j < num_medoids; j++){
        d_clusterDistances[j] = d_distaneMatrix[i*N + medoids[j]];
    }

    return findSmallestandSecondSmallest(d_clusterDistances, num_medoids);
}

__global__ void DECalculator(int* d_medoids, size_t num_medoids, float* d_ds, float* d_es, float* d_distanceMatrix, int N){
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < N){
        float* res = findDE(i, d_medoids, num_medoids, d_distanceMatrix, N);
        d_ds[i] = res[0];
        d_es[i] = res[1];
        cudaFree(res);
    }
}

std::pair<float*, float*> DECalculatorWrapper(std::unordered_set<int>& medoids, float* d_distanceMatrix, int N){
    // allocate memory
    int* h_medoids = set_to_array(medoids);
    int* d_medoids;
    float *d_ds, *d_es;
    cudaMalloc((void**)&d_medoids, medoids.size()*sizeof(int));
    cudaMalloc((void**)&d_ds, N*sizeof(float));
    cudaMalloc((void**)&d_es, N*sizeof(float));

    // copy to device
    cudaMemcpy(d_medoids, h_medoids, medoids.size()*sizeof(int), cudaMemcpyHostToDevice);

    // launch the kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1)/blockSize;
    DECalculator<<<numBlocks, blockSize>>>(d_medoids, medoids.size(), d_ds, d_es, d_distanceMatrix, N);

    // free memeory
    delete[] h_medoids;
    cudaFree(d_medoids);

    cudaDeviceSynchronize();
    return std::make_pair(d_ds, d_es);
}


__global__ void TihCalculator(int* d_candidates, size_t num_candidates, int* d_medoids, size_t num_medoids, float* d_Tih, float* d_ds, float* d_es, float* d_distanceMatrix, int N) {
    int h = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.y + blockDim.y*blockIdx.y;
    if(h < num_candidates && i < num_medoids){
        d_Tih[h*num_medoids + i] = 0;
        for(int j = 0; j < num_candidates; j++){
            if(d_candidates[j] == d_candidates[h]) continue;
            if(d_distanceMatrix[d_candidates[j]*N + d_candidates[i]] > d_ds[d_candidates[j]]){
                d_Tih[h*num_medoids + i] += std::min(d_distanceMatrix[d_candidates[j]*N + d_candidates[h]] - d_ds[d_candidates[j]], (float)0);
            }
            else{
                d_Tih[h*num_medoids + i] += std::min(d_distanceMatrix[d_candidates[j]*N + d_candidates[h]], d_es[d_candidates[j]]) - d_ds[d_candidates[j]];
            }
        }
    }
}

float* TihCalculatorWrapper(int* h_medoids, size_t num_medoids, int* h_candidates, size_t num_candidates, float* d_ds, float* d_es, float* d_distanceMatrix, int N){
    // allocate the memory
    int *d_medoids, *d_candicates;
    float* h_res, *d_res;
    h_res = new float[num_medoids*num_candidates];
    cudaMalloc((void**)&d_medoids, num_medoids*sizeof(int));
    cudaMalloc((void**)&d_candicates, num_candidates*sizeof(int));
    cudaMalloc((void**)&d_res, num_medoids*num_candidates*sizeof(float));


    // copying to device
    cudaMemcpy(d_medoids, h_medoids, num_medoids*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candicates, h_candidates, num_candidates*sizeof(int), cudaMemcpyHostToDevice);

    // Launching the kernel
    dim3 numBlocks(256, 256);
    dim3 blockSize((num_candidates + blockSize.x - 1)/blockSize.x, (num_medoids + blockSize.y -1)/blockSize.y);
    TihCalculator<<<numBlocks, blockSize>>>(d_candicates, num_candidates, d_medoids, num_medoids, d_res, d_ds, d_es, d_distanceMatrix, N);
    
    // Copying result to host
    cudaMemcpy(h_res, d_res, num_candidates*num_medoids*sizeof(float), cudaMemcpyDeviceToHost);

    // Free Unused data
    cudaFree(d_medoids);
    cudaFree(d_candicates);
    cudaFree(d_res);

    cudaDeviceSynchronize();
    return h_res;
}