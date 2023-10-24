#include<data.hpp>
#include<pam.hpp>
#include<iostream>


int main(){
    // (1, 2), (2, 2), (2, 3), (3, 3), (8, 7), (7, 8), (9, 7), (8, 8)
    // (0, 0), (0, 1), (1, 0), (1, 1), (4, 4), (4, 5), (5, 4), (5, 5)
    std::vector<Point> points = {
    Point{0, 0},
    Point{0, 1},
    Point{1, 0},
    Point{1, 1},
    Point{40, 40},
    Point{45, 50},
    Point{51, 40},
    Point{55, 45},
    Point{-40, -50},
    Point{-45, -51},
    Point{-43, -55},
    // Point{4, 2},
    // Point{4, 3},
    // Point{5, 1},
    // Point{5, 2},
    // Point{5, 3},
    // Point{6, 1},
    // Point{6, 2},
    // Point{6, 3},
    // Point{7, 1}
};



    auto data = new SimpleData(points);
    auto algo = new PAM(data, 3);
    // std::cout << "Starting build\n";
    algo->build();
    std::cout << "Initial medoids ";
    for(auto i : algo->medoids){
        std::cout << i << " ";
    }
    std::cout << std::endl;

    algo->swap();
    std::cout << "Final medoids ";
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