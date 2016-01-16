package com.skiwi.neuralnetwork

import java.util.function.DoubleUnaryOperator

/**
 * @author Frank van Heeswijk
 */
class Neuron {
    List<NeuronConnection> inputs = []
    List<NeuronConnection> outputs = []

    double value
    double delta

    void initializeWeights(Random random) {
        inputs.each { it.weight = random.nextDouble() }
    }

    double getInputSum() {
        inputs.stream().mapToDouble({ it.inValue }).sum()
    }

    double getWeightedOutputDelta() {
        outputs.stream().mapToDouble({ it.outDelta }).sum()
    }

    void forwardPropagate(DoubleUnaryOperator activationFunction) {
        value = activationFunction.applyAsDouble(inputSum)
    }

    void calculateDeltaValue(DoubleUnaryOperator activationDerivativeFunction) {
        if (!outputs) {
            throw new IllegalStateException("This is an output neuron")
        }
        delta = activationDerivativeFunction.applyAsDouble(inputSum) * weightedOutputDelta
    }

    void calculateDeltaValue(DoubleUnaryOperator activationDerivativeFunction, double error) {
        if (outputs) {
            throw new IllegalStateException("This is not an output neuron")
        }
        delta = activationDerivativeFunction.applyAsDouble(inputSum) * error
    }

    void updateWeights(double learningRate) {
        inputs.each { it.updateWeight(learningRate) }
    }
}
