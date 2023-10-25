#pragma once

#include<string>
#include<cmath>
#include<vector>
#include<fstream>
#include<sstream>
#include<iostream>

class Point{
    public:
        float x, y;

        Point() = default;

        Point(float x, float y){
            this->x = x;
            this->y = y;
        }

        float getEucledianDist(Point p){
            return sqrt((x-p.x)*(x-p.x) + (y-p.y)*(y-p.y));
        }
};

float getEucledianDist(Point a, Point b){
    return a.getEucledianDist(b);
}

class SimpleData{
    
    public:
        SimpleData(std::vector<Point>& points){
            this->points = std::move(points);
            allocateDistanceMatrixGPU();
        }

        SimpleData(std::string filename){
            std::ifstream csv(filename);
            std::vector<Point> pts;

            if(csv.is_open()){
                std::string line;
                while(std::getline(csv, line)){
                    std::vector<std::string> res;
                    std::stringstream ss(line);
                    std::string item;

                    while(getline(ss, item, ',')){
                        res.push_back(item);
                    }

                    pts.push_back(Point{std::stof(res[0]), std::stof(res[1])});
                }

                this->points = std::move(pts);
            }
            else{
                std::cerr << "Could not open the csv\n";
            }

            allocateDistanceMatrixGPU();
        }

        ~SimpleData(){
            cudaFree(d_distMat);
        }

        float getEucledianDist(int i, int j){
            return points[i].getEucledianDist(points[j]);
        }

        Point getithPoint(int i){
            return points[i];
        }

        int getSize(){return points.size();}

        float* getDeviceDistMat(){return d_distMat;}


    private:
        std::vector<Point> points;
        float* d_distMat;

        void initDistanceMatrix(float*& h_matrix, int N){
            h_matrix = new float[N*N];
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    h_matrix[i*N+j] = getEucledianDist(i, j);
                }
            }

            // std::cout << "Distance matrix on Host initialized\n";
            // for(int i = 0; i < N; i++){
            //     for(int j = 0; j < N; j++){
            //         std::cout << h_matrix[i*N+j] << " ";
            //     }
            //     std::cout << std::endl;
            // } 
        }

        void allocateDistanceMatrixGPU(){
            int N = getSize();
            float* h_matrix, *d_matrix;

            initDistanceMatrix(h_matrix, N);

            cudaMalloc((void**)&d_matrix, N*N*sizeof(float));
            cudaMemcpy(d_matrix, h_matrix, N*N*sizeof(float), cudaMemcpyHostToDevice);
            delete[] h_matrix;
            this->d_distMat = d_matrix;
        }

};
