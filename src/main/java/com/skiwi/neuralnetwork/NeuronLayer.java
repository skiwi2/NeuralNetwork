package com.skiwi.neuralnetwork;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

import static com.skiwi.neuralnetwork.LayerType.INPUT;
import static com.skiwi.neuralnetwork.LayerType.OUTPUT;

/**
 * @author Frank van Heeswijk
 */
public class NeuronLayer {
    private final LayerType layerType;
    private final Neuron[] neurons;

    private NeuronLayer previousLayer;
    private NeuronLayer nextLayer;

    private NeuronLayer(LayerType layerType, Neuron[] neurons) {
        this.layerType = layerType;
        this.neurons = neurons;
    }

    public LayerType getLayerType() {
        return layerType;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    public void setPreviousLayer(NeuronLayer previousLayer) {
        this.previousLayer = previousLayer;
    }

    public NeuronLayer getPreviousLayer() {
        return previousLayer;
    }

    public void setNextLayer(NeuronLayer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public NeuronLayer getNextLayer() {
        return nextLayer;
    }

    void initializeNeuronWeights(Random random) {
        for (Neuron neuron : neurons) {
            neuron.initializeWeights(random);
        }
    }

    void setQueryVector(double[] queryVector) {
        if (layerType != INPUT) {
            throw new IllegalStateException("This is not an input neuron layer");
        }
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].setValue(queryVector[i]);
        }
    }

    void forwardPropagate(DoubleUnaryOperator activationFunction) {
        for (Neuron neuron : neurons) {
            neuron.forwardPropagate(activationFunction);
        }
    }

    double[] calculateErrorVector(double[] targetVector) {
        if (layerType != OUTPUT) {
            throw new IllegalStateException("This is not an output neuron layer");
        }
        double[] errorVector = new double[targetVector.length];
        for (int i = 0; i < targetVector.length; i++) {
            errorVector[i] = targetVector[i] - neurons[i].getValue();
        }
        return errorVector;
    }

    void calculateDeltaValues(DoubleUnaryOperator activationDerivativeFunction) {
        if (layerType == OUTPUT) {
            throw new IllegalStateException("This is an output neuron layer");
        }
        for (Neuron neuron : neurons) {
            neuron.calculateDeltaValue(activationDerivativeFunction);
        }
    }

    void calculateDeltaValues(DoubleUnaryOperator activationDerivativeFunction, double[] targetVector) {
        if (layerType != OUTPUT) {
            throw new IllegalStateException("This is not an output neuron layer");
        }
        double[] errorVector = calculateErrorVector(targetVector);
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].calculateDeltaValue(activationDerivativeFunction, errorVector[i]);
        }
    }

    void updateWeights(double learningRate) {
        for (Neuron neuron : neurons) {
            neuron.updateWeights(learningRate);
        }
    }

    double[] getOutputVector() {
        if (layerType != OUTPUT) {
            throw new IllegalStateException("This is not an output neuron layer");
        }
        double[] outputVector = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputVector[i] = neurons[i].getValue();
        }
        return outputVector;
    }

    void addBiasNode(Neuron biasNeuron) {
        for (Neuron neuron : neurons) {
            neuron.addInput(new NeuronConnection(biasNeuron, neuron));
        }
    }

    static NeuronLayer create(LayerType layerType, int neuronCount) {
        Neuron[] neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron();
        }
        return new NeuronLayer(layerType, neurons);
    }

    static void connect(NeuronLayer first, NeuronLayer second) {
        for (Neuron n1 : first.getNeurons()) {
            for (Neuron n2 : second.getNeurons()) {
                NeuronConnection neuronConnection = new NeuronConnection(n1, n2);
                n1.addOutput(neuronConnection);
                n2.addInput(neuronConnection);
            }
        }
        first.setNextLayer(second);
        second.setPreviousLayer(first);
    }
}
