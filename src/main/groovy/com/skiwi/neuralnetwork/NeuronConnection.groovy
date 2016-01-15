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
}
