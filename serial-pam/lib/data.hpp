#pragma once

#include<string>
#include<cmath>
#include<vector>

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

        float getEucledianDist(int i, int j){
            return points[i].getEucledianDist(points[j]);
        }

        int getSize(){return points.size();}


    private:
        std::vector<Point> points;


};
