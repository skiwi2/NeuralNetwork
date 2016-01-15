package com.skiwi.neuralnetwork

/**
 * @author Frank van Heeswijk
 */
interface NeuralNetwork {
    void learn(double learningRate)

    double[] query(double[] queryVector)
}