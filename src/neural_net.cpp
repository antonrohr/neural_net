//
//  neural_net.cpp
//  neural_net_cpp
//
//  Created by Anton Rohr on 16.04.19.
//

#include "neural_net.hpp"


using namespace std;

NeuralNet::NeuralNet(const int inputLength, const int hiddenLayer, const int hiddenLayerSize, const int outputLength)
: epochs(0)
, outputLayer(outputLength, hiddenLayerSize)
{
    //hiddenLayer
    int previousSize = inputLength;
    for (int i = 0; i < hiddenLayer; i++) {
        hiddenLayers.emplace_back(HiddenLayer(hiddenLayerSize, previousSize));
        previousSize = hiddenLayerSize;
    }
}

NeuralNet::NeuralNet(string filePath, const int inputLength)
{
    
    // file to vector of strings
    ifstream file(filePath);
    string text( (istreambuf_iterator<char>(file)), (istreambuf_iterator<char>()) );
    istringstream stream(text);
    string line;
    vector<string> layersWeights;
    while (getline(stream, line)) {
        if (line.rfind("epochs", 0) != string::npos) {
            string val = line.substr(7);
            cout << "epochs found; val: " << val << endl;
            epochs = stoi(val);
        } else if (line == "layer") {
            layersWeights.emplace_back("");
        } else {
            layersWeights[layersWeights.size() - 1] += line + "\n";
        }
    }
    
    for (int i = 0; i < layersWeights.size() - 1; i++) {
        hiddenLayers.emplace_back(HiddenLayer(layersWeights[i]));
    }
    
    outputLayer = OutputLayer(layersWeights[layersWeights.size() - 1]);
    
//    cout << text;
    // TODO
    
}

double NeuralNet::computeError(const vector<double>& input, const uint8_t& label)
{

    vector<double> results = forwardPropagate(input);
    double result = 0;

    for (uint8_t i = 0; i < 10; i++) {
        double tmp = results[i];
        if (i == label) {
            tmp -= 1.0;
        }
        result += pow(tmp, 2);
    }

    return result;
}

uint8_t NeuralNet::predict(const vector<double> &input)
{
    
    vector<double> results = forwardPropagate(input);
    
    int maxIndex = max_element(results.begin(), results.end()) - results.begin();
    
    return (uint8_t) maxIndex;
    
}

double NeuralNet::trainEpoch(const vector<vector<double> > &input, const vector<uint8_t> &label, const double learningRate) {

    assert(input.size() == label.size());

    double accumulatedError = 0.0;
    
    for (int i = 0; i < input.size(); i++) {

        accumulatedError += computeError(input[i], label[i]);
        backwardPropagate(label[i]);
        updateWeights(input[i], learningRate);

    }
    
    epochs++;

    return accumulatedError / input.size();

}

void NeuralNet::writeToFile(string filePath) {
    
    string text = "epochs " + to_string(epochs) + "\n";
    
    for (int i = 0; i < hiddenLayers.size(); i++) {
        text += hiddenLayers[i].serialize();
    }
    
    text += outputLayer.serialize();
    
    ofstream file;
    file.open(filePath);
    file << text;
    file.close();
    
}

vector<double> NeuralNet::forwardPropagate(const vector<double>& inputs)
{
    
    vector<double> results = inputs;
    
    for (int i = 0; i < hiddenLayers.size(); i++) {
        results = hiddenLayers[i].forwardPropagate(results);
    }
    
    results = outputLayer.forwardPropagate(results);
    
    return results;
}

void NeuralNet::backwardPropagate(vector<double>& expected)
{
    
    outputLayer.backwardPropagate(expected);
    
    Layer* nextLayer = &outputLayer;
    
    for (int i = hiddenLayers.size() - 1; i >= 0; i--) {
        
        hiddenLayers[i].backwardPropagate(*nextLayer);
        
        nextLayer = &hiddenLayers[i];
        
    }
}

void NeuralNet::backwardPropagate(uint8_t expected)
{
    
    vector<double> expectedValues;
    for (int i = 0; i < 10; i++) {
        if (i == (int) expected) {
            expectedValues.push_back(1.0);
        } else {
            expectedValues.push_back(0.0);
        }
    }
    
    backwardPropagate(expectedValues);
}

void NeuralNet::updateWeights(const vector<double>& input, const double learningRate)
{
    
    vector<double> inputs = input;
    
    for (int i = 0; i < hiddenLayers.size(); i++) {
        inputs = hiddenLayers[i].updateWeights(inputs, learningRate);
    }
    
    outputLayer.updateWeights(inputs, learningRate);
    
}

int NeuralNet::getEpochs() const
{
    return epochs;
}

double NeuralNet::test(const vector<vector<double> > &testImages, vector<uint8_t>& testLabels)
{
    assert(testImages.size() == testLabels.size());
    
    int wrongCounter = 0;
    
    for (int i = 0; i < testImages.size(); i++) {
        
        uint8_t prediction = predict(testImages[i]);
        
        if (prediction != testLabels[i]) {
            wrongCounter++;
        }
        
    }
    
    return (double) wrongCounter * 100 / testImages.size();
    
}
