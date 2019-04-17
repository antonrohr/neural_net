//
//  neural_net.cpp
//  neural_net_cpp
//
//  Created by Anton Rohr on 16.04.19.
//

#include "neural_net.hpp"



using namespace std;

NeuralNet::NeuralNet()
: NeuralNet(784, 2, 16, 10)
{
    cout << "Neural Net initialized!" << endl;
}

NeuralNet::NeuralNet(int inputLength, int hiddenLayer, int hiddenLayerSize, int outputLength)
{

    // input Layer
    layers.emplace_back(make_unique<InputLayer>(inputLength));
    
    // hidden Layer
    int previousSize = inputLength;
    for (int i = 0; i < hiddenLayer; i++) {
        layers.emplace_back(make_unique<Layer>(hiddenLayerSize, previousSize));
        previousSize = hiddenLayerSize;
    }
    
    // outputLayer
    layers.emplace_back(make_unique<Layer>(outputLength, previousSize));
}

vector<double> NeuralNet::compute(std::vector<double> inputs){
    
    vector<double> results = inputs;
    
    for(unique_ptr<Layer> const& layerPtr: layers) {
        results = layerPtr->compute(results);
        
        cout << "layer results: ";
        for (double result: results) {
            cout << result << " ";
        }
        cout << endl;
    }
    
    return results;
}
