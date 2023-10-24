#pragma once
#include<data.hpp>
#include<unordered_set>
#include<limits>
#include<iostream>

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
            // std::cout << "N is" << N << std::endl;
            int first_medoid = -1;
            float dist = std::numeric_limits<float>::max();
            for(int i = 0; i < N; i++){
                float dist_sum = 0;
                for(int j = 0; j < N; j++){
                    dist_sum += data->getEucledianDist(i,j);
                }
                if(dist_sum < dist){
                    first_medoid = i;
                    dist = dist_sum;
                }
            }
            // std::cout << first_medoid << std::endl;

            medoids.insert(first_medoid);
            candidates.erase(first_medoid);

            while(medoids.size() != k){
                // if(medoid_candidates.size() > 0) std::cout << *medoid_candidates.begin() << std::endl;
                int final_candidate = -1;
                float max_gain = -1 * std::numeric_limits<float>::max();
                for(auto i : candidates){
                    float gain = 0;
                    for(auto j : candidates){
                        if(j == i) continue;
                        float dissimilariy = std::numeric_limits<float>::max();
                        for(auto medoid: medoids) dissimilariy = std::min(dissimilariy, data->getEucledianDist(j, medoid));
                        gain += std::max(dissimilariy - data->getEucledianDist(j, i), (float)0);
                    }

                    if(gain > max_gain){
                        std::cout << gain << std::endl;
                        max_gain = gain;
                        final_candidate = i;
                    }

                }

                medoids.insert(final_candidate);
                candidates.erase(final_candidate);
            }
        }        

};