#include<data.hpp>
#include<pam.hpp>
#include<iostream>


int main(){
    std::vector<Point> points = {
    Point{1, 1},
    Point{1, 2},
    Point{1, 3},
    Point{2, 1},
    Point{0, -1},
    Point{2, 2},
    Point{2, 3},
    Point{3, 1},
    Point{3, 2},
    Point{3, 3},
    Point{4, 1},
    Point{4, 2},
    Point{4, 3},
    Point{5, 1},
    Point{5, 2},
    Point{5, 3},
    Point{6, 1},
    Point{6, 2},
    Point{6, 3},
    Point{7, 1}
};

    for (int i = 0; i < 200; ++i) {
        Point p;
        p.x = static_cast<double>(rand()) / RAND_MAX * 100.0;  // Adjust 10.0 for desired x range
        p.y = static_cast<double>(rand()) / RAND_MAX * 100.0;  // Adjust 10.0 for desired y range
        points.push_back(p);
    }


    auto data = new SimpleData(points);
    auto algo = new PAM(data, 10);
    // std::cout << "Starting build\n";
    algo->build();
    for(auto i : algo->medoids){
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // std::unordered_set<int> s;
    // s.insert(1);
    // s.insert(2);
    // s.insert(3);
    // s.insert(4);

    // for(auto i : s){
    //     for(auto j : s){
    //         std::cout << i << " " << j << std::endl;
    //     }
    // }
}