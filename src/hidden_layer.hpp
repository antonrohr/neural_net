//
//  hidden_layer.hpp
//  neural_net
//
//  Created by Anton Rohr on 06.05.19.
//

#ifndef hidden_layer_hpp
#define hidden_layer_hpp

#include <stdio.h>

#include "layer.hpp"

class HiddenLayer: public Layer {
protected:
//    auto activationFunction(double input) -> double;
//    auto transferDerivative(double output) -> double;
public:
    HiddenLayer(const int size, const int previousSize);
    HiddenLayer(const std::string layerData);
    std::vector<double> forwardPropagate(const std::vector<double>& values);
    void backwardPropagate(const Layer& nextLayer);
};

#endif /* hidden_layer_hpp */
