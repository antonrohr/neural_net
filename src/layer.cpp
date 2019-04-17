//
//  layer.cpp
//  neural_net
//
//  Created by Anton Rohr on 16.04.19.
//

#include "layer.hpp"


using namespace std;

Layer::Layer(int size, int previousSize) {
    for (int i = 0; i < size; i++) {
        nodes.push_back(Node(previousSize));
    }
}

Layer::~Layer()
{}

vector<double> Layer::compute(std::vector<double> values) {
    vector<double> results;
    
    for (int i = 0; i < nodes.size(); i++) {
        results.push_back(nodes[i].compute(values));
    }
    
    return results;
}

void Layer::print() {
    
    cout << "Layer with " << nodes.size() << " nodes: " << endl;
    for (Node node: nodes) {
        cout << " - ";
        node.print();
    }
}
