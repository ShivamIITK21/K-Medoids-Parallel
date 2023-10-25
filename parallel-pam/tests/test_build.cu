#include<data.cuh>
#include<pam.cuh>
#include<chrono>
#include<string>

int main(int argc, char* argv[]){
    if(argc != 3){
        std::cerr << "Usage ./<executable> <csv path> <k>\n";
    }
    auto data = new SimpleData(argv[1]);
    auto algo = new PAM(data, std::stoi(argv[2]));
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