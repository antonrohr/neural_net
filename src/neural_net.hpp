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
#include <fstream>
#include <sstream>

//#include "node.hpp"
#include "layer.hpp"
#include "input_layer.hpp"

class NeuralNet {
    std::vector<std::unique_ptr<Layer>> layers;
    
    double computeError(const std::vector<double>& img, const uint8_t& label) const;
    void train(const std::vector<double>& img, const uint8_t& label);
    void adjustWeights();
public:
    NeuralNet(const int inputLength, const int hiddenLayer, const int hiddenLayerSize, const int outputLength);
    NeuralNet(const std::string filePath, const int inputLength);
    std::vector<double> compute(const std::vector<double>& inputs) const;
    uint8_t predict(const std::vector<double>& img) const;
    void train(const std::vector<std::vector<double>>& images, const std::vector<uint8_t>& label, const int& batchSize);
    double computeAvgError(const std::vector<std::vector<double>>& images, const std::vector<uint8_t>& label);
    void writeToFile(std::string filePath);
};


#endif /* neural_net_hpp */
