//
//  node.cpp
//  neural_net
//
//  Created by Anton Rohr on 16.04.19.
//

#include "node.hpp"

using namespace std;

//Node::Node(vector<double> weights, double bias)
//: weights(weights)
//, bias(bias)
//, accWeightNudges(weights.size(), 0.0)
//, accBiasNudge(0.0)
//, trainingCounter(0)
//{}

Node::Node(int numberOfWeights)
: accWeightNudges(numberOfWeights, 0.0)
, accBiasNudge(0.0)
, trainingCounter(0)
{
    for (int i = 0; i < numberOfWeights; i++) {
        weights.push_back(getRandom());
    }
    bias = getRandom();
}

double sumOfProducts(const vector<double>& a, const vector<double>& b) {
    
    assert(a.size() == b.size());
    
    double result = 0;
    for(int i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}


double Node::compute(const std::vector<double>& values) const
{
    double result = sumOfProducts(weights, values) + bias;
    return activationFunction(result);
}

void Node::print() const
{
    cout << "Node with " << weights.size() << " weights: ";
    for(double weight: weights) {
        cout << weight << " ";
    }
    cout << endl;
}

double Node::activationFunction(double input){
    return 1.0 / (1.0 + exp(-input));
//    return max(0.0, input);
}

double Node::derivedActivationFunction(double input){
    return exp(-input) / pow(1 + exp(-input), 2.0);
//    return input > 0 ? 1.0 : 0.0;
}

double Node::getRandom(){
    static mt19937 mt(1.0);
    static uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(mt);
}

vector<double> Node::train(const vector<double>& previousLayerValues, const double& wantedResult)
{
    
    double zL = sumOfProducts(weights, previousLayerValues) + bias;
    
    double aL = activationFunction(zL);
    
    // dC0/daL
    double wantedDerivedByAL = 2 * (aL - wantedResult);
    
    // daL/dzL
    double aLDerivedByZL = derivedActivationFunction(zL);
    
    // dzL/dwL
    vector<double> zLDerivedByWL = previousLayerValues;
    
    // weights
    vector<double> weightNudge(zLDerivedByWL.size());
    transform(zLDerivedByWL.begin(), zLDerivedByWL.end(), weightNudge.begin(), [&](double value) -> double {
        return value * wantedDerivedByAL * aLDerivedByZL;
    });
    // save Weights
    for (int i = 0; i < accWeightNudges.size(); i++) {
        accWeightNudges[i] += weightNudge[i];
    }
    
    // bias
    double biasNudge = wantedDerivedByAL + aLDerivedByZL;
    // save Bias
    accBiasNudge += biasNudge;
    
    trainingCounter++;
    
    // previousValues
    vector<double> previousValuesNudges = vector<double>(weights.size());
    transform(weights.begin(), weights.end(), previousValuesNudges.begin(), [&](double weight) -> double {
        return weight * aLDerivedByZL * wantedDerivedByAL;
    });
    
    return previousValuesNudges;
}

void Node::adjustWeights(){
    
    for (int i = 0; i < weights.size(); i++) {
        weights[i] -= accWeightNudges[i] / trainingCounter;
    }
    bias -= accBiasNudge / trainingCounter;
    
    trainingCounter = 0;
    accBiasNudge = 0;
    accWeightNudges = vector<double>(weights.size(), 0.0);
    
}
