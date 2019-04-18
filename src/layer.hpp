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

#include "node.hpp"

class Layer {
protected:
    std::vector<Node> nodes;
public:
    Layer(int size, int previousSize);
    virtual ~Layer();
    virtual std::vector<double> compute(const std::vector<double>& values) const;
    void print() const;
    std::vector<double> train(const std::vector<double>& previousLayerValues, const std::vector<double>& wantedValues);
    void adjustWeights();
    std::string serialize();
};

#endif /* layer_hpp */
