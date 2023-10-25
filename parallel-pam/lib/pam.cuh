#pragma once
#include<data.cuh>
#include<unordered_set>
#include<limits>
#include<iostream>
#include<utility>
#include<kernels.cuh>

class PAM{

    public:
        PAM(SimpleData* data, int k){
            this->data = data;
            this->k = k;
            for(int i = 0; i < data->getSize(); i++) candidates.insert(i);
        }

    // private:
        std::unordered_set<int> medoids;
        std::unordered_set<int> candidates;
        SimpleData* data;
        int k;

        void build(){
            int N = data->getSize();

            //selecting the first medoid
            int first_medoid = -1;
            float dist = std::numeric_limits<float>::max();

            //serial
            // for(int i = 0; i < N; i++){
            //     float dist_sum = 0;
            //     for(int j = 0; j < N; j++){
            //         dist_sum += data->getEucledianDist(i,j);
            //     }
            //     if(dist_sum < dist){
            //         first_medoid = i;
            //         dist = dist_sum;
            //     }
            // }

            // parallel
            float* res = distanceSumAllPointsWrapper(N, data->getDeviceDistMat());
            for(int i = 0; i < N; i++){
                std::cout << res[i] << " ";
            }

            // medoids.insert(first_medoid);
            // candidates.erase(first_medoid);

            // selecting the remaining medoids
            // while(medoids.size() != k){
            //     int final_candidate = -1;
            //     float max_gain = -1 * std::numeric_limits<float>::max();
            //     for(auto i : candidates){
            //         float gain = 0;
            //         for(auto j : candidates){
            //             if(j == i) continue;
            //             float dissimilariy = std::numeric_limits<float>::max();
            //             for(auto medoid: medoids) dissimilariy = std::min(dissimilariy, data->getEucledianDist(j, medoid));
            //             gain += std::max(dissimilariy - data->getEucledianDist(j, i), (float)0);
            //         }

            //         if(gain > max_gain){
            //             std::cout << gain << std::endl;
            //             max_gain = gain;
            //             final_candidate = i;
            //         }

            //     }

            //     medoids.insert(final_candidate);
            //     candidates.erase(final_candidate);
            // }
        }        

        std::pair<float, float> findSmallestandSecondSmallest(std::vector<float>& v){
            int idx = -1;
            float smallest = std::numeric_limits<float>::max();
            for(int i = 0; i < v.size(); i++){
                if(v[i] < smallest){
                    smallest = v[i];
                    idx = i;
                }
            }

            float secondSmallest = std::numeric_limits<float>::max();
            for(int i = 0; i < v.size(); i++){
                if(i == idx) continue;
                if(v[i] < secondSmallest) secondSmallest = v[i];
            }

            return std::make_pair(smallest, secondSmallest);
        }

        std::pair<float, float> findDE(int i){
            std::vector<float> clusterDistances;
            for(auto medoid : medoids){
                clusterDistances.push_back(data->getEucledianDist(i, medoid));
            }
            return findSmallestandSecondSmallest(clusterDistances);
        }

        void swap(){
            auto curr_medoids = medoids;
            int it = 0;
            while(1){
                // calculating the Dj and Ej for every j
                std::cout << ++it << "swap iteration" << std::endl;
                int N = data->getSize();
                std::vector<float> ds(N), es(N);
                for(int i = 0; i < N; i++){
                    auto de = findDE(i);
                    ds[i] = de.first;
                    es[i] = de.second;
                }

                // calculating swap cost for every pair
                int mincost_h = -1;
                int mincost_i = -1;
                float mincost = std::numeric_limits<float>::max(); 
                for(auto h : candidates){
                    for(auto i : medoids){
                        float Tih = 0;
                        for(auto j : candidates){
                            if(j == h) continue;
                            if(data->getEucledianDist(j, i) > ds[j]){
                                Tih += std::min(data->getEucledianDist(j, h) - ds[j], (float)0);
                            }
                            else{
                                Tih += std::min(data->getEucledianDist(j, h), es[j]) - ds[j];
                            }
                        }
                        if(Tih < mincost){
                            mincost = Tih;
                            mincost_h = h;
                            mincost_i = i;
                        }
                    }
                }

                if(mincost < 0){
                    candidates.erase(mincost_h);
                    medoids.erase(mincost_i);
                    candidates.insert(mincost_i);
                    medoids.insert(mincost_h);
                }
                else{
                    break;
                }
            }
        }

};