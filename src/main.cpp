#include <iostream>
// #include <thread>
#include <vector>

#include "neural_net.hpp"

using namespace std;

double getRandom(){
    static mt19937 mt(1.0);
    static uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(mt);
}

int main(){
	
    NeuralNet neuralNet = NeuralNet();
    
    vector<double> input;
    for (int i = 0; i < 784; i++) {
        input.push_back(getRandom());
    }
    
    vector<double> results = neuralNet.compute(input);
    
    cout << "end results: ";
    for(double result: results) {
        cout << result << " ";
    }
    cout << endl;
    
}
