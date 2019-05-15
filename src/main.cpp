#include <iostream>
// #include <thread>
#include <vector>

#include "neural_net.hpp"
#include "node.hpp"
#include "mnist/mnist_reader.hpp"

using namespace std;

void normalizeImages(const vector<vector<uint8_t>>& images, vector<vector<double>>& out) {
    
    assert(images.size() == out.size());
    
    for (int i = 0; i < images.size(); i++) {
        
        const vector<uint8_t>& imgRaw = images[i];
        vector<double>& img = out[i];
        
        transform(imgRaw.begin(), imgRaw.end(), img.begin(),
                  [](uint8_t val) -> double { return val / 255.0; });

    }
}

string newFilePath() {
    
    string filePath = "/Users/antonrohr/git/neural_net/data/weights_";
    
    time_t currentMS = time(nullptr);
    
    return filePath + to_string(currentMS) + ".txt";
    
}


void method2() {
    
    string filePath = "/Users/antonrohr/git/neural_net/data/new_weights.txt";
    
    NeuralNet nn = NeuralNet(10, 3, 2, 10);
//    nn.writeToFile(filePath);

//    NeuralNet nn = NeuralNet(filePath, 10);
    
    //generate training Data:
    int numberTrainingExamples = 10000;
    vector<vector<double>> trainingData;
    vector<uint8_t> trainingLabels;
    
    for (int i = 0; i < numberTrainingExamples; i++) {
        int randInt = Utilities::getRandomInt(0, 9);
    
        vector<double> data;
        for(int j = 0; j < 10; j++){
            if (j == randInt) {
                data.push_back(1.0);
            } else {
                data.push_back(0.0);
            }
        }
        
        trainingData.push_back(data);
        trainingLabels.push_back((uint8_t) randInt);
    }
    
    //generate test Data
    int numberTestExamples = 1000;
    vector<vector<double>> testData;
    vector<uint8_t> testLabels;
    
    for (int i = 0; i < numberTestExamples; i++) {
        
        int randInt = Utilities::getRandomInt(0, 9);
        
        vector<double> data;
        for(int j = 0; j < 10; j++){
            if (j == randInt) {
                data.push_back(1.0);
            } else {
                data.push_back(0.0);
            }
        }
        
        testData.push_back(data);
        testLabels.push_back((uint8_t) randInt);
        
    }
    
    
    const double learningRate = 0.01;
    const int epochs = 1000;
    
    for (int i = 0; i < epochs; i++) {
        double error = nn.trainEpoch(trainingData, trainingLabels, learningRate);
        cout << "epoch: " << nn.getEpochs() << " avg epoch error: " << error << endl;
        
        nn.writeToFile(filePath);
        
        double errorRateTrainingSet = nn.test(trainingData, trainingLabels);
        cout << "Error Rate on Training Data: " << errorRateTrainingSet << "%" << endl;
        
        double errorRateTestSet = nn.test(testData, testLabels);
        cout << "Error Rate on Test Data: " << errorRateTestSet << "%" << endl;
    }
}


int main(){
    
    method2();
    return 0;
    
    string filePath = "/Users/antonrohr/git/neural_net/data/weights.txt";
    
//    NeuralNet neuralNet = NeuralNet(filePath, 784);
    
    NeuralNet neuralNet = NeuralNet(784, 2, 16, 10);
//    neuralNet.writeToFile(filePath);
//    return 0;
    
    mnist::MNIST_dataset<vector, vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<vector, vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    
    typedef vector<double> image;
    
    // training test set
    vector<image> trainingImages(dataset.training_images.size(), image(784));
    normalizeImages(dataset.training_images, trainingImages);
    vector<uint8_t>& trainingLabels = dataset.training_labels;
    
    // prediction test set
    vector<image> testImages(dataset.test_images.size(), image(784));
    normalizeImages(dataset.test_images, testImages);
    vector<uint8_t>& testLabels = dataset.test_labels;
    
    double learningRate = 0.1;
    int epochs = 1000;
    
    for (int j = 0; j < epochs; j++) {
        
        double error = neuralNet.trainEpoch(trainingImages, trainingLabels, learningRate);
        cout << "epoch: " << neuralNet.getEpochs() << " avg epoch error: " << error << endl;
        neuralNet.writeToFile(filePath);
    
    
        double errorRateTrainingSet = neuralNet.test(trainingImages, trainingLabels);
        cout << "Error Rate on Training Data: " << errorRateTrainingSet << "%" << endl;
        
        double errorRateTestSet = neuralNet.test(testImages, testLabels);
        cout << "Error Rate on Test Data: " << errorRateTestSet << "%" << endl;
    
    }
}
