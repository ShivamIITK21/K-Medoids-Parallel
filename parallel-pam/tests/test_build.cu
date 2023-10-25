#include<data.cuh>
#include<pam.cuh>
#include<chrono>

int main(){
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

    auto data = new SimpleData("./data/data_biggest.csv");
    auto algo = new PAM(data, 11);
    auto start = std::chrono::high_resolution_clock::now();
    algo->build();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Build took " << duration.count() << std::endl;
    std::cout << "Initial medoids ";
    for(auto i : algo->medoids){
        std::cout << i << " ";
    }
    std::cout << std::endl;

    start = std::chrono::high_resolution_clock::now();    
    algo->swap();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Swap took " << duration.count() << std::endl;
    std::cout << "Final medoids ";
    for(auto i : algo->medoids){
        std::cout << i << " ";
    }
    std::cout << std::endl;

    for(auto i : algo->medoids){
        std::cout << data->getithPoint(i).x << " " << data->getithPoint(i).y << std::endl;
    }
}