//
//  node.cpp
//  neural_net
//
//  Created by Anton Rohr on 16.04.19.
//

#include "node.hpp"

using namespace std;

Node::Node(vector<double> weights, double bias)
: weights(weights)
, bias(bias)
{}

Node::Node(int numberOfWeights)
{
    for (int i = 0; i < numberOfWeights; i++) {
        weights.push_back(getRandom());
    }
    bias = getRandom();
}

double Node::compute(vector<double> values)
{
    if (values.size() != weights.size()) {
        return -100;
    }
    double result = 0;
    for (int i = 0; i < weights.size(); i++) {
        result += weights[i] * values[i];
    }
    return activationFunction(result + bias);
}

void Node::print()
{
    cout << "Node with " << weights.size() << " weights: ";
    for(double weight: weights) {
        cout << weight << " ";
    }
    cout << endl;
}

double Node::activationFunction(double input){
    return 1.0 / (1.0 + exp(-input));
}

double Node::getRandom(){
    static mt19937 mt(1.0);
    static uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(mt);
}
