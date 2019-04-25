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
    string weights;
    
    while (getline(layerStream , weights)) {
        
        istringstream weightsStream(weights);
        string weight;
        
        vector<double> weightsData;
        
        while (getline(weightsStream, weight, ' ')) {
            weightsData.emplace_back(stod(weight));
        }
        
        nodes.emplace_back(Node(weightsData));
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
            accResult[i] += result[i];
        }
    }
    
    for (int i = 0; i < accResult.size(); i++) {
        accResult[i] /= nodes.size();
    }
    
    return accResult;
    
}

void Layer::adjustWeights() {
    
    for (Node& node: nodes) {
        node.adjustWeights();
    }
    
}

string Layer::serialize() const
{
    string text = "layer\n";
    for (int i = 0; i < nodes.size(); i++) {
        text += nodes[i].serialize();
    }
    
    return text;
}
