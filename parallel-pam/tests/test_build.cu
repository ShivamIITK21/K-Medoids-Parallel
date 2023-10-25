#include<data.cuh>
#include<pam.cuh>

int main(){
    std::vector<Point> points = {
    Point{0, 0},
    Point{0, 1},
    Point{1, 0},
    Point{1, 1},
    Point{40, 40},
//     Point{45, 50},
//     Point{51, 40},
//     Point{55, 45},
//     Point{-40, -50},
//     Point{-45, -51},
//     Point{-43, -55},
//     // Point{4, 2},
//     // Point{4, 3},
//     // Point{5, 1},
//     // Point{5, 2},
//     // Point{5, 3},
//     // Point{6, 1},
//     // Point{6, 2},
//     // Point{6, 3},
//     // Point{7, 1}
    };

    auto data = new SimpleData(points);
    auto algo = new PAM(data, 2);
    algo->build();
}