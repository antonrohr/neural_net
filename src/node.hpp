//
//  node.hpp
//  neural_net
//
//  Created by Anton Rohr on 16.04.19.
//

#ifndef node_hpp
#define node_hpp

#include <iostream>
#include <vector>
#include <cmath>

#include "utilities.hpp"

class Node {
    std::vector<double> weights;
    double bias;
    std::vector<double> accWeightNudges;
    double accBiasNudge;
    double output;
    int trainingCounter;
    static double activationFunction(double input);
    static double derivedActivationFunction(double input);
    
public:
    double delta;
    Node(const std::vector<double> weights, const double bias);
    Node(const int numberOfWeights);
    double compute(const std::vector<double>& values) const;
    void print() const;
    std::vector<double> train(const std::vector<double>& previousLayerValues, const double& wantedResult);
    void adjustWeights();
    std::string serialize() const;
    double getOutput() const;
    double forwardPropagate(const std::vector<double>& inputs, auto activationFunction(double input) -> double);
    double getWeight(int index) const;
    void updateWeights(const std::vector<double>& input, const double learningRate);
};

#endif /* node_hpp */
