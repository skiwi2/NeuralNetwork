package com.skiwi.neuralnetwork

import java.util.function.DoubleUnaryOperator

import static com.skiwi.neuralnetwork.LayerType.INPUT
import static com.skiwi.neuralnetwork.LayerType.OUTPUT

/**
 * @author Frank van Heeswijk
 */
class NeuronLayer {
    LayerType layerType
    Neuron[] neurons

    NeuronLayer previousLayer
    NeuronLayer nextLayer

    NeuronLayer(LayerType layerType, Neuron[] neurons) {
        this.layerType = layerType
        this.neurons = neurons
    }

    void initializeNeuronWeights(Random random) {
        neurons.each { it.initializeWeights(random) }
    }

    void setQueryVector(double[] queryVector) {
        if (layerType != INPUT) {
            throw new IllegalStateException("This is not an input neuron layer")
        }
        neurons.eachWithIndex{ neuron, i -> neuron.value = queryVector[i] }
    }

    void forwardPropagate(DoubleUnaryOperator activationFunction) {
        neurons.each { it.forwardPropagate(activationFunction) }
    }

    double[] calculateErrorVector(double[] targetVector) {
        if (layerType != OUTPUT) {
            throw new IllegalStateException("This is not an output neuron layer")
        }
        def errorVector = new double[targetVector.length]
        for (int i = 0; i < targetVector.length; i++) {
            errorVector[i] = targetVector[i] - neurons[i].value
        }
        errorVector
    }

    void calculateDeltaValues(DoubleUnaryOperator activationDerivativeFunction) {
        if (layerType == OUTPUT) {
            throw new IllegalStateException("This is an output neuron layer")
        }
        neurons.each { it.calculateDeltaValue(activationDerivativeFunction) }
    }

    void calculateDeltaValues(DoubleUnaryOperator activationDerivativeFunction, double[] targetVector) {
        if (layerType != OUTPUT) {
            throw new IllegalStateException("This is not an output neuron layer")
        }
        def errorVector = calculateErrorVector(targetVector)
        neurons.eachWithIndex{ neuron, i -> neuron.calculateDeltaValue(activationDerivativeFunction, errorVector[i]) }
    }

    void updateWeights(double learningRate) {
        neurons.each { it.updateWeights(learningRate) }
    }

    double[] getOutputVector() {
        if (layerType != OUTPUT) {
            throw new IllegalStateException("This is not an output neuron layer")
        }
        Arrays.stream(neurons).mapToDouble({ it.value }).toArray()
    }

    void addBiasNode(Neuron biasNeuron) {
        neurons.each { it.inputs << new NeuronConnection(biasNeuron, it) }
    }

    static NeuronLayer create(LayerType layerType, int neuronCount) {
        def neurons = new Neuron[neuronCount]
        neurons.eachWithIndex{ n, i -> neurons[i] = new Neuron() }
        new NeuronLayer(layerType, neurons)
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
