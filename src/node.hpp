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
#include <random>

class Node {
    std::vector<double> weights;
    double bias;
    std::vector<double> accWeightNudges;
    double accBiasNudge;
    int trainingCounter;
    static double activationFunction(double input);
    static double derivedActivationFunction(double input);
    static double getRandom();
public:
//    Node(std::vector<double> weights, double bias);
    Node(int numberOfWeights);
    double compute(const std::vector<double>& values) const;
    void print() const;
    std::vector<double> train(const std::vector<double>& previousLayerValues, const double& wantedResult);
    void adjustWeights();
    std::string serialize();
};

#endif /* node_hpp */
