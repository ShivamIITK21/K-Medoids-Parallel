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
        }

        float getEucledianDist(int i, int j){
            return points[i].getEucledianDist(points[j]);
        }

        Point getithPoint(int i){
            return points[i];
        }

        int getSize(){return points.size();}


    private:
        std::vector<Point> points;


};
