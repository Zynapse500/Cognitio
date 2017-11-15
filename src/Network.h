//
// Created by zynapse on 14/11/17.
//

#ifndef PROJECT_COGNITIO_NETWORK_H
#define PROJECT_COGNITIO_NETWORK_H

#include <vector>
#include <random>
#include <ctime>

using std::move;



struct Node {
    std::vector<double> weights;
    double bias;

    explicit Node(unsigned inputs);
};



typedef std::vector<Node> Layer;



class Network {

    std::vector<Layer> layers;

public:

    explicit Network(std::vector<unsigned> shape);

    std::vector<double> feed(std::vector<double> input);

    double train(std::vector<double> input, std::vector<double> target, double learningRate);

private:

    std::vector<std::vector<double>> forwardPass(std::vector<double> input);
    double ReLU(double x);
    double dReLU(double x);
};

#endif //PROJECT_COGNITIO_NETWORK_H
