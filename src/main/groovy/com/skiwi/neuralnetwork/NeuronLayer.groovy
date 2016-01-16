package com.skiwi.neuralnetwork

import java.lang.reflect.Array
import java.util.function.DoubleUnaryOperator

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

    void initializeNeuronWeights(Random random) {
        neurons.each { it.initializeWeights(random) }
    }

    //TODO only allow this is if the current NeuronLayer is the input layer
    void setQueryVector(double[] queryVector) {
        neurons.eachWithIndex{ neuron, i -> neuron.value = queryVector[i] }
    }

    void forwardPropagate(DoubleUnaryOperator activationFunction) {
        neurons.each { it.forwardPropagate(activationFunction) }
    }

    //TODO only allow this if the current NeuronLayer is the output layer
    double[] calculateErrorVector(double[] targetVector) {
        def errorVector = new double[targetVector.length]
        for (int i = 0; i < targetVector.length; i++) {
            errorVector[i] = targetVector[i] - neurons[i].value
        }
        errorVector
    }

    void calculateDeltaValues(DoubleUnaryOperator activationDerivativeFunction) {
        //TODO check if current NeuronLayer is the output layer instead of performing this calculation
        def outputNeuronCount = Arrays.stream(neurons).mapToInt({ it.outputs.size() }).sum()
        if (!outputNeuronCount) {
            throw new IllegalStateException("This is an output neuron layer")
        }
        neurons.each { it.calculateDeltaValue(activationDerivativeFunction) }
    }

    void calculateDeltaValues(DoubleUnaryOperator activationDerivativeFunction, double[] targetVector) {
        //TODO check if current NeuronLayer is the output layer instead of performing this calculation
        def outputNeuronCount = Arrays.stream(neurons).mapToInt({ it.outputs.size() }).sum()
        if (outputNeuronCount) {
            throw new IllegalStateException("This is not an output neuron layer")
        }
        def errorVector = calculateErrorVector(targetVector)
        neurons.eachWithIndex{ neuron, i -> neuron.calculateDeltaValue(activationDerivativeFunction, errorVector[i]) }
    }

    void updateWeights(double learningRate) {
        neurons.each { it.updateWeights(learningRate) }
    }

    //TODO only allow this is if the current NeuronLayer is the output layer
    double[] getOutputVector() {
        Arrays.stream(neurons).mapToDouble({ it.value }).toArray()
    }

    void addBiasNode(Neuron biasNeuron) {
        neurons.each { it.inputs << new NeuronConnection(biasNeuron, it) }
    }

    static NeuronLayer create(int neuronCount) {
        def neurons = new Neuron[neuronCount]
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
