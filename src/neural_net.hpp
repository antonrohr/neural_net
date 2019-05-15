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
#include "hidden_layer.hpp"
#include "output_layer.hpp"

class NeuralNet {
    int epochs;
    OutputLayer outputLayer;
    std::vector<HiddenLayer> hiddenLayers;
public:
    NeuralNet(const int inputLength, const int hiddenLayer, const int hiddenLayerSize, const int outputLength);
    NeuralNet(const std::string filePath, const int inputLength);
    double computeError(const std::vector<double>& img, const uint8_t& label);
    uint8_t predict(const std::vector<double>& img);
    double trainEpoch(const std::vector<std::vector<double>>& images, const std::vector<uint8_t>& label, const double learningRate);
    void writeToFile(std::string filePath);
    std::vector<double> forwardPropagate(const std::vector<double>& inputs);
    void backwardPropagate(std::vector<double>& expected);
    void backwardPropagate(uint8_t expected);
    void updateWeights(const std::vector<double>& input, const double learningRate);
    int getEpochs() const;
    double test(const std::vector<std::vector<double>>& testImages, std::vector<uint8_t>& testLabels);
};


#endif /* neural_net_hpp */
