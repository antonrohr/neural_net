//
//  output_layer.cpp
//  neural_net
//
//  Created by Anton Rohr on 26.04.19.
//

#include "output_layer.hpp"

using namespace std;

OutputLayer::OutputLayer(const int size, const int previousSize)
: Layer(size, previousSize)
{}

OutputLayer::OutputLayer(const string layerData)
: Layer(layerData)
{}

OutputLayer::OutputLayer()
: Layer(0,0)
{}

vector<double> OutputLayer::forwardPropagate(const vector<double>& values)
{
 
    vector<double> result(nodes.size());
    
    transform(nodes.begin(), nodes.end(), result.begin(), [&](Node& node){
        return node.forwardPropagate(values, Utilities::sigmoid);
    });
    
    return result;
}

void OutputLayer::backwardPropagate(const vector<double> &expected)
{
    for (int i = 0; i < nodes.size(); i++) {
        nodes[i].delta = (expected[i] - nodes[i].getOutput()) * Utilities::sigmoidTransferDerivative(nodes[i].getOutput());
    }
}
