//
//  utilities.hpp
//  neural_net
//
//  Created by Anton Rohr on 25.04.19.
//

#ifndef utilities_hpp
#define utilities_hpp

#include <stdio.h>
#include <random>
#include <sstream>

class Utilities {
    static double alpha;
    static int randomSeed;
public:
    static double getRandomDouble(const double min, const double max);
    static int getRandomInt(const double min, const double max);
    static double sigmoid(double input);
    static double sigmoidTransferDerivative(double output);
    static double reLU(double input);
    static double reLUTransferDerivative(double output);
    static double leakyReLU(double input);
    static double leakyReLUTransferDerivative(double output);
    static double tanh(double input);
    static double tanhTransferDerivative(double output);
    static std::string doubleToString(double input);
};

#endif /* utilities_hpp */
