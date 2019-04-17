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
    static double activationFunction(double input);
    static double getRandom();
public:
    Node(std::vector<double> weights, double bias);
    Node(int numberOfWeights);
    Node();
    double compute(std::vector<double> values);
    void print();
};

#endif /* node_hpp */
