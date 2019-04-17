//
//  input_layer.hpp
//  neural_net
//
//  Created by Anton Rohr on 16.04.19.
//

#ifndef input_layer_hpp
#define input_layer_hpp

#include <iostream>
#include <vector>

#include "layer.hpp"

class InputLayer: public Layer {
public:
    InputLayer(int size);
    std::vector<double> compute(std::vector<double> values);
};

#endif /* input_layer_hpp */