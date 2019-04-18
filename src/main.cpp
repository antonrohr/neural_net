#include <iostream>
// #include <thread>
#include <vector>

#include "neural_net.hpp"
#include "mnist/mnist_reader.hpp"

using namespace std;

double getRandom(){
    static mt19937 mt(1.0);
    static uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(mt);
}

void imgNormalize(const vector<uint8_t>& img, vector<double>& out) {
    
    transform(img.begin(), img.end(), out.begin(),
              [](uint8_t val) -> double { return val / 255.0; });
    
}

int main(){
	
    NeuralNet neuralNet = NeuralNet(784, 2, 16, 10);
    
    mnist::MNIST_dataset<vector, vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<vector, vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    
    vector<double> img(784);
    
    
    
    
    
    
    int runs = dataset.training_images.size();

    double accError = 0;
    for (int i = 0; i < runs; i++) {
        imgNormalize(dataset.training_images[i], img);
        uint8_t& label = dataset.training_labels[i];
        accError += neuralNet.computeError(img, label);
    }
    cout << "AvgError before: " << accError / runs << endl;

    for (int i = 0; i < runs; i++) {
        if (i != 0 && i % 100 == 0) {
            neuralNet.adjustWeights();
        }
        
        imgNormalize(dataset.training_images[i], img);
        uint8_t& label = dataset.training_labels[i];
        neuralNet.train(img, label);
        
    }

    neuralNet.adjustWeights();

    accError = 0.0;
    for (int i = 0; i < runs; i++) {
        imgNormalize(dataset.training_images[i], img);
        uint8_t& label = dataset.training_labels[i];
        accError += neuralNet.computeError(img, label);
    }
    cout << "AvgError after: " << accError / runs << endl;
    
    
    
    // prediction test set
    int predRuns = dataset.test_images.size();
    int counterWrong = 0;
    accError = 0;
    
    for (int i = 0; i < predRuns; i++) {
        
        imgNormalize(dataset.test_images[i], img);
        uint8_t& label = dataset.test_labels[i];
        
        uint8_t prediction = neuralNet.predict(img);
        
        if (label != prediction) {
            counterWrong++;
//            cout << "label: " << (int) label << " prediction: " << (int) prediction << endl;
        }
        
        accError += neuralNet.computeError(img, label);
        
    }
    
    cout << predRuns << " predictions Runs. Of those " << counterWrong << " wrongly predicted" << endl;
    cout << "average prediction error: " << accError / predRuns << endl;
    
}
