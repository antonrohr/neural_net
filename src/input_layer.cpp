//
//  input_layer.cpp
//  neural_net
//
//  Created by Anton Rohr on 16.04.19.
//

#include "input_layer.hpp"

using namespace std;

InputLayer::InputLayer(int size)
: Layer(0, 1)
{}

vector<double> InputLayer::compute(std::vector<double> values) {
    return values;
}