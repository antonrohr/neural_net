//
//  neural_net.hpp
//  neural_net_cpp
//
//  Created by Anton Rohr on 16.04.19.
//

#ifndef neural_net_hpp
#define neural_net_hpp

#include <iostream>
#include <vector>

//#include "node.hpp"
#include "layer.hpp"
#include "input_layer.hpp"

class NeuralNet {
    std::vector<std::unique_ptr<Layer>> layers;
public:
    NeuralNet(int inputLength, int hiddenLayer, int hiddenLayerSize, int outputLength);
    std::vector<double> compute(const std::vector<double>& inputs) const;
    double computeError(const std::vector<double>& img, const uint8_t& label) const;
    void train(const std::vector<double>& img, const uint8_t& label);
    void adjustWeights();
    uint8_t predict(const std::vector<double>& img) const;
};


#endif /* neural_net_hpp */
