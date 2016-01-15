package com.skiwi.neuralnetwork

import java.lang.reflect.Array

/**
 * @author Frank van Heeswijk
 */
class NeuronLayer {
    Neuron[] neurons

    NeuronLayer previousLayer
    NeuronLayer nextLayer

    NeuronLayer(Neuron[] neurons) {
        this.neurons = neurons
    }

    static NeuronLayer create(int neuronCount) {
        Neuron[] neurons = new Neuron[neuronCount]
        neurons.eachWithIndex{ n, i -> neurons[i] = new Neuron() }
        new NeuronLayer(neurons)
    }

    static void connect(NeuronLayer first, NeuronLayer second) {
        first.neurons.each { n1 -> second.neurons.each { n2 ->
            def connection = new NeuronConnection(n1, n2)
            n1.outputs << connection
            n2.inputs << connection
        } }
        first.nextLayer = second
        second.previousLayer = first
    }
}
