package com.skiwi.neuralnetwork
/**
 * @author Frank van Heeswijk
 */
interface NeuralNetwork {
    void learn(LearningData learningData, double learningRate)

    double[] query(double[] queryVector)
}