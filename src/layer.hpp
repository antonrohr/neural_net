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
    virtual std::vector<double> compute(std::vector<double> values);
    void print();
};

#endif /* layer_hpp */