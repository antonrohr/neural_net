//
//  output_layer.hpp
//  neural_net
//
//  Created by Anton Rohr on 26.04.19.
//

#ifndef output_layer_hpp
#define output_layer_hpp

#include <stdio.h>

#include "layer.hpp"

class OutputLayer: public Layer {
public:
    OutputLayer(const int size, const int previousSize);
    OutputLayer(const std::string layerData);
    OutputLayer();
    std::vector<double> forwardPropagate(const std::vector<double>& values);
    void backwardPropagate(const std::vector<double>& expected);
};

#endif /* output_layer_hpp */
