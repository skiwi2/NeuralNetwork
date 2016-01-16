package com.skiwi.neuralnetwork

/**
 * @author Frank van Heeswijk
 */
class NeuronConnection {
    Neuron input
    Neuron output

    double weight

    NeuronConnection(Neuron input, Neuron output) {
        this.input = input
        this.output = output
    }

    double getInValue() {
        weight * input.value
    }

    double getOutValue() {
        weight * output.value
    }

    double getOutDelta() {
        weight * output.delta
    }

    void updateWeight(double learningRate) {
        weight += learningRate * input.value * output.delta
    }
}
