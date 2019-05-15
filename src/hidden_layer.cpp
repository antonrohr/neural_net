//
//  hidden_layer.cpp
//  neural_net
//
//  Created by Anton Rohr on 06.05.19.
//

#include "hidden_layer.hpp"

using namespace std;

HiddenLayer::HiddenLayer(const int size, const int previousSize)
: Layer(size, previousSize)
{}

HiddenLayer::HiddenLayer(const string layerData)
: Layer(layerData)
{}

//double HiddenLayer::activationFunction(double input)
//{
//    return Utilities::reLU(input);
//}
//
//double HiddenLayer::transferDerivative(double output)
//{
//    return Utilities::reLUTransferDerivative(output);
//}

vector<double> HiddenLayer::forwardPropagate(const vector<double>& values)
{
    
    vector<double> result(nodes.size());
    
    transform(nodes.begin(), nodes.end(), result.begin(), [&](Node& node){
        return node.forwardPropagate(values, Utilities::leakyReLU);
    });
    
    return result;
}

void HiddenLayer::backwardPropagate(const Layer& nextLayer)
{
    
    for (int i = 0; i < nodes.size(); i++) {
        
        double error = 0.0;
        for (int j = 0; j < nextLayer.nodes.size(); j++) {
            const Node& nodeInner = nextLayer.nodes[j];
            error += nodeInner.getWeight(i) * nodeInner.delta;
        }
        
        Node& nodeOuter = nodes[i];
        
        nodeOuter.delta = error * Utilities::leakyReLUTransferDerivative(nodeOuter.getOutput());
    }
    
}
