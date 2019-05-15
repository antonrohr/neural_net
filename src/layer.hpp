//
//  layer.hpp
//  neural_net
//
//  Created by Anton Rohr on 16.04.19.
//

#ifndef layer_hpp
#define layer_hpp

#include <iostream>
#include <vector>
#include <sstream>


#include "node.hpp"

class Layer {
protected:
//    static auto activationFunction(double input) -> double;
//    static auto transferDerivative(double output) -> double;
public:
    std::vector<Node> nodes;
    Layer(const int size, const int previousSize);
    Layer(const std::string layerData);
    virtual ~Layer();
    virtual std::vector<double> compute(const std::vector<double>& values) const;
    void print() const;
    std::vector<double> train(const std::vector<double>& previousLayerValues, const std::vector<double>& wantedValues);
    virtual std::string serialize() const;
    virtual std::vector<double> forwardPropagate(const std::vector<double>& values) = 0;
    std::vector<double> updateWeights(const std::vector<double>& input, const double learningRate);
};

#endif /* layer_hpp */
