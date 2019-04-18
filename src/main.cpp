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

void normalizeImages(const vector<vector<uint8_t>>& images, vector<vector<double>>& out) {
    
    assert(images.size() == out.size());
    
    for (int i = 0; i < images.size(); i++) {
        
        const vector<uint8_t>& imgRaw = images[i];
        vector<double>& img = out[i];
        
        transform(imgRaw.begin(), imgRaw.end(), img.begin(),
                  [](uint8_t val) -> double { return val / 255.0; });
        
    }
    
    
    
}

int main(){
	
    NeuralNet neuralNet = NeuralNet(784, 2, 16, 10);
    
    mnist::MNIST_dataset<vector, vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<vector, vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    
    typedef vector<double> image;
    
    vector<image> trainingImages(dataset.training_images.size(), image(784));
    normalizeImages(dataset.training_images, trainingImages);
    
    vector<uint8_t>& trainingLabels = dataset.training_labels;
    
    string filePath = "/Users/antonrohr/git/neural_net/data/weights.txt";
    neuralNet.writeToFile(filePath);
    neuralNet.readFromFile(filePath);

    return 0;
    
    double avgError = neuralNet.computeAvgError(trainingImages, trainingLabels);
    
    
    cout << "AvgError before: " << avgError << endl;

    neuralNet.train(trainingImages, trainingLabels, 1000);
    
    
    
    avgError = neuralNet.computeAvgError(trainingImages, trainingLabels);
    
    cout << "AvgError after: " << avgError << endl;



    // prediction test set
    vector<image> testImages(dataset.test_images.size(), image(784));
    normalizeImages(dataset.test_images, testImages);
    vector<uint8_t>& testLabels = dataset.test_labels;
    
    int counterWrong = 0;

    for (int i = 0; i < testImages.size(); i++) {

        uint8_t prediction = neuralNet.predict(testImages[i]);

        if (testLabels[i] != prediction) {
            counterWrong++;
        }

    }

    cout << testImages.size() << " predictions Runs. Of those " << counterWrong << " wrongly predicted" << endl;
    avgError = neuralNet.computeAvgError(testImages, testLabels);
    cout << "average prediction error: " << avgError << endl;
    
}
