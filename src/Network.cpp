//
// Created by zynapse on 14/11/17.
//

#include <stdexcept>
#include "Network.h"


Node::Node(unsigned inputs) {
    this->weights.resize(inputs, 1.0);
}

Network::Network(std::vector<unsigned> shape) {
    unsigned long layerCount = shape.size();

    if(layerCount < 2) {
        throw std::runtime_error("Network must contain at least 2 layers!");
    }

    auto lastSize = shape[0];
    for (int i = 1; i < layerCount; ++i) {
        auto size = shape[i];
        this->layers.emplace_back(size, Node(lastSize));
        lastSize = size;
    }

    // Initialize weights to random values
    std::default_random_engine rng((unsigned long)(time(0)));

    layerCount = layers.size();
    for (int i = 0; i < layerCount; ++i) {
        auto nodeCount = layers[i].size();
        for (int j = 0; j < nodeCount; ++j) {

            auto n = shape[i];
            double num = sqrt(2.0 / n);
            std::uniform_real_distribution<double> distribution(0.0, n);

            auto weightCount = layers[i][j].weights.size();
            for (int k = 0; k < weightCount; ++k) {
                layers[i][j].weights[k] = distribution(rng) * num;
            }

            layers[i][j].bias = 0;
        }
    }
}

std::vector<double> Network::feed(std::vector<double> input) {
    return move(forwardPass(move(input)).back());
}

double Network::train(std::vector<double> input, std::vector<double> target, double learningRate) {
    double errorTotal = 0;
    auto outputs = forwardPass(move(input));
    auto adjustedLayers = layers;

    std::vector<double> prevDeltas;

    auto layerCount = layers.size();
    for (long i = layerCount - 1; i >= 0; --i) {
        std::vector<double> deltas;

        auto nodeCount = layers[i].size();
        for (int j = 0; j < nodeCount; ++j) {
            // Perform the delta rule
            double delta = dReLU(outputs[i + 1][j]);

            // Output layer
            if (i == layerCount - 1) {
                // Calculate the error of the node and the network
                double diff = target[j] - outputs[i + 1][j];
                double error = (diff * diff) / 2;
                errorTotal += error;

                delta *=-diff;
            } else {
                double prevDeltaSum = 0;
                auto prevDeltaCount = prevDeltas.size();
                for (int l = 0; l < prevDeltaCount; ++l) {
                    // Calculate the gradient of each weight
                    auto weight = layers[i + 1][l].weights[j];
                    prevDeltaSum += prevDeltas[l] * weight;
                }
                delta *= prevDeltaSum;
            }


            // Calculate the gradient of each weight
            auto weightCount = layers[i][j].weights.size();
            for (int k = 0; k < weightCount; ++k) {
                double gradient = learningRate * delta * outputs[i][k];
                adjustedLayers[i][j].weights[k] -= gradient;
            }

            deltas.push_back(delta);
        }

        prevDeltas = move(deltas);
    }

    layers = move(adjustedLayers);

    return errorTotal;
}


std::vector<std::vector<double>> Network::forwardPass(std::vector<double> input) {
    // Forward pass
    //  - Store all layer's activations/outputs
    std::vector<std::vector<double>> outputs = {move(input)};

    for (auto &&layer : layers) {
        std::vector<double> output;

        auto& prevOutput = outputs.back();
        for (auto &&node : layer) {
            double result = 0;

            unsigned windex = 0;
            for (auto &&weight : node.weights) {
                result += prevOutput[windex] * weight;
                windex++;
            }

            output.push_back(ReLU(result+ node.bias));
        }

        outputs.push_back(move(output));
    }

    return move(outputs);
}

double Network::ReLU(double x) {
    if (x < 0) return 0;
    return x;
}

double Network::dReLU(double x) {
    if (x < 0) return 0;
    return 1;
}
