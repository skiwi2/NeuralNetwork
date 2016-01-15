package com.skiwi.neuralnetwork

/**
 * @author Frank van Heeswijk
 */
class SimpleNeuralNetwork implements NeuralNetwork {
    NeuronLayer[] layers

    SimpleNeuralNetwork(int inputNeuronCount, int outputNeuronCount, int... hiddenNeuronCount) {
        generateNetwork(inputNeuronCount, outputNeuronCount, hiddenNeuronCount)
    }

    private void generateNetwork(int inputNeuronCount, int outputNeuronCount, int... hiddenNeuronCount) {
        layers = [inputNeuronCount, *hiddenNeuronCount, outputNeuronCount].stream()
            .map({ NeuronLayer.create(it) })
            .toArray({ i -> new NeuronLayer[i] })
        for (int i = 0; i < layers.length - 1; i++) {
            NeuronLayer.connect(layers[i], layers[i + 1])
        }
    }

    @Override
    void learn(LearningData learningData, double learningRate) {
        throw new UnsupportedOperationException()
    }

    @Override
    double[] query(double[] queryVector) {
        throw new UnsupportedOperationException()
    }
}
