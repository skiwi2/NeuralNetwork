package com.skiwi.neuralnetwork;

import java.util.function.DoubleUnaryOperator;

/**
 * @author Frank van Heeswijk
 */
public interface NeuralNetwork {
    void learn(LearningData learningData, double learningRate, DoubleUnaryOperator activationFunction, DoubleUnaryOperator activationDerivativeFunction, int maxIterations);

    double[] query(double[] queryVector, DoubleUnaryOperator activationFunction);
}