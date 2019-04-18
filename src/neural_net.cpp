//
//  neural_net.cpp
//  neural_net_cpp
//
//  Created by Anton Rohr on 16.04.19.
//

#include "neural_net.hpp"



using namespace std;

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

vector<double> NeuralNet::compute(const vector<double>& inputs) const
{
    
    vector<double> results = inputs;
    
    for(const unique_ptr<Layer>& layerPtr: layers) {
        results = layerPtr->compute(results);
    }
    
    return results;
}

double NeuralNet::computeError(const vector<double>& img, const uint8_t& label) const
{
    
    vector<double> results = compute(img);
    double result = 0;
    
    for (int i = 0; i < 10; i++) {
        double tmp = results[i];
        if (i == label) {
            tmp -= 1.0;
        }
        result += pow(tmp, 2);
    }
    
    return result;
}

void NeuralNet::train(const vector<double> &img, const uint8_t& label)
{
    
    vector<vector<double>> savedResults;
    
    for (int i = 0; i < layers.size(); i++) {
        Layer& layer = *layers[i];
        
        if (i == 0) {
            savedResults.push_back(layer.compute(img));
        } else {
            savedResults.push_back(layer.compute(savedResults.back()));
        }
    }
    
    vector<double> wantedResults;
    for (uint8_t i = 0; i < 10; i++) {
        if (i == label) {
            wantedResults.push_back(1.0);
        } else {
            wantedResults.push_back(0.0);
        }
    }
    
    for (int i = layers.size() - 1; i > 0 ; --i) {
        Layer& layer = *layers[i];
        
        assert(savedResults[i].size() == wantedResults.size());
        
        wantedResults = layer.train(savedResults[i-1], wantedResults);
        
    }
    
}

void NeuralNet::adjustWeights()
{
    
    for (int i = 1; i < layers.size(); i++) {
        
        Layer& layer = *layers[i];
        
        layer.adjustWeights();
    }
    
}

uint8_t NeuralNet::predict(const vector<double> &img) const
{
    
    vector<double> results = compute(img);
    
    int maxIndex = max_element(results.begin(), results.end()) - results.begin();
    
    return (uint8_t) maxIndex;
    
}

void NeuralNet::train(const vector<vector<double> > &images, const vector<uint8_t> &label, const int &batchSize) {
    
    assert(images.size() == label.size());
    
    for (int i = 0; i < images.size(); i++) {
        
        if (i % batchSize == 0){
            adjustWeights();
        }
        
        train(images[i], label[i]);
        
    }
    
    adjustWeights();
    
}

double NeuralNet::computeAvgError(const vector<vector<double> > &images, const vector<uint8_t> &label)
{
    
    assert(images.size() == label.size());
    
    double accError = 0;
    
    for (int i = 0; i < images.size(); i++) {
        accError += computeError(images[i], label[i]);
    }
    
    return accError / images.size();
    
}

void NeuralNet::writeToFile(string filePath) {
    
    string text;
    
    for (int i = 0; i < layers.size(); i++) {
        Layer& layer = *layers[i];
        text += layer.serialize();
    }
    
    ofstream file;
    file.open(filePath);
    file << text;
    file.close();
    
}

void NeuralNet::readFromFile(string filePath)
{
    ifstream file(filePath);
    string text( (istreambuf_iterator<char>(file)), (istreambuf_iterator<char>()) );
    
    cout << text;
    // TODO
}
