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
    NeuralNet();
    NeuralNet(int inputLength, int hiddenLayer, int hiddenLayerSize, int outputLength);
    std::vector<double> compute(std::vector<double> inputs);
};


#endif /* neural_net_hpp */
