//
//  utilities.cpp
//  neural_net
//
//  Created by Anton Rohr on 25.04.19.
//

#include "utilities.hpp"

using namespace std;

int Utilities::randomSeed = 4.0;

double Utilities::alpha = 0.01;

double Utilities::getRandomDouble(const double min, const double max) {
    
    static mt19937 mt(randomSeed);
    static uniform_real_distribution<double> dist(min, max);
    return dist(mt);
    
}

int Utilities::getRandomInt(const double min, const double max)
{
    static mt19937 mt(randomSeed);
    static uniform_int_distribution<int> dist(min, max);
    return dist(mt);
}

double Utilities::sigmoid(double input)
{
    return 1.0 / (1.0 + exp(-input));
}

double Utilities::sigmoidTransferDerivative(double output)
{
    return output * (1.0 - output);
}

double Utilities::reLU(double input)
{
    return max(0.0, input);
}

double Utilities::reLUTransferDerivative(double output)
{
    return output > 0 ? 1 : 0;
}

double Utilities::leakyReLU(double input)
{
    return (input > 0) ? input : input * alpha;
}

double Utilities::leakyReLUTransferDerivative(double output)
{
    return (output > 0) ? 1 : alpha;
}

double Utilities::tanh(double input)
{
    return std::tanh(input);
//    return (exp(2 * input) - 1) / (exp(2 * input) + 1);
}

double Utilities::tanhTransferDerivative(double output)
{
    return 1.0 - pow(output, 2.0);
}

string Utilities::doubleToString(double input) {
    ostringstream out;
    out.precision(12);
    out << fixed << input;
    return out.str();
}
