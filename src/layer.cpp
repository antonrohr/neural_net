//
//  layer.cpp
//  neural_net
//
//  Created by Anton Rohr on 16.04.19.
//

#include "layer.hpp"


using namespace std;

Layer::Layer(const int size, const int previousSize) {
    for (int i = 0; i < size; i++) {
        nodes.emplace_back(Node(previousSize));
    }
}

Layer::Layer(const string layerData) {
    
    istringstream layerStream(layerData);
    string weightsString;
    
    while (getline(layerStream , weightsString)) {
        
        istringstream weightsStream(weightsString);
        
        // bias
        string biasString;
        getline(weightsStream, biasString, ' ');
        double bias = stod(biasString);
        
        // weights
        string weightString;
        vector<double> weights;
        
        while (getline(weightsStream, weightString, ' ')) {
            weights.emplace_back(stod(weightString));
        }
        
        nodes.emplace_back(Node(weights, bias));
    }
    
}

Layer::~Layer()
{}

vector<double> Layer::compute(const std::vector<double>& values) const
{
    vector<double> results;
    
    for (int i = 0; i < nodes.size(); i++) {
        results.push_back(nodes[i].compute(values));
    }
    
    return results;
}

void Layer::print() const {
    
    cout << "Layer with " << nodes.size() << " nodes: " << endl;
    for (Node node: nodes) {
        cout << " - ";
        node.print();
    }
}

vector<double> Layer::train(const std::vector<double>& previousLayerValues, const std::vector<double>& wantedValues) {
    
    assert(wantedValues.size() == nodes.size());
    
    vector<double> accResult(previousLayerValues.size());
    
    for (int i = 0; i < nodes.size(); i++) {
        vector<double> result = nodes[i].train(previousLayerValues, wantedValues[i]);
        
        assert(result.size() == accResult.size());
        
        for (int j = 0; j < result.size(); j++) {
            accResult[j] += result[j];
        }
    }
    
    for (int i = 0; i < accResult.size(); i++) {
        accResult[i] /= nodes.size();
    }
    
    return accResult;
    
}

string Layer::serialize() const
{
    string text = "layer\n";
    for (int i = 0; i < nodes.size(); i++) {
        text += nodes[i].serialize();
    }
    
    return text;
}

//vector<double> Layer::forwardPropagate(const vector<double>& values)
//{
//    
//    vector<double> result(nodes.size());
//
//    transform(nodes.begin(), nodes.end(), result.begin(), [&](Node& node) {
//        return node.forwardPropagate(values, activationFunction);
//    });
//    
//    return result;
//    
//}

//double Layer::activationFunction(double input)
//{
//    return Utilities::sigmoid(input);
//}
//
//double Layer::transferDerivative(double output)
//{
//    return Utilities::sigmoidTransferDerivative(output);
//}

vector<double> Layer::updateWeights(const vector<double>& input, const double learningRate)
{
    
    vector<double> result;
    
    for (int i = 0; i < nodes.size(); i++) {
        result.push_back(nodes[i].getOutput());
        nodes[i].updateWeights(input, learningRate);
    }
    
    return result;
}

