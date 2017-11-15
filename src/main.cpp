#include <iostream>
#include <vector>

#include <random>

#include "Utility.h"

#include "Network.h"


int main() {
    std::default_random_engine rng;

    Network net({2, 5, 1});

    std::vector<double> input = {1, 1};
    auto output = net.feed(input);
    std::cout << input << " -> " << output << std::endl;


    // Train on XOR
    double errorSum = 0;
    for (int i = 0; i < 1000000; ++i) {
        auto a = bool(rng() % 2);
        auto b = bool(rng() % 2);

        auto c = bool(a ^ b);

        errorSum += net.train({double(a), double(b)}, {double(c)}, 5e-3);

        const int N = 10000;
        if (i % N == 0) {
            std::cout << "Error: " << errorSum / N << std::endl;
            errorSum = 0;
        }
    }
    std::cout << "\n";

    input = {0, 0};
    output = net.feed(input);
    std::cout << input << " -> " << output << std::endl;

    input = {1, 0};
    output = net.feed(input);
    std::cout << input << " -> " << output << std::endl;

    input = {0, 1};
    output = net.feed(input);
    std::cout << input << " -> " << output << std::endl;

    input = {1, 1};
    output = net.feed(input);
    std::cout << input << " -> " << output << std::endl;


	return 0;
}
