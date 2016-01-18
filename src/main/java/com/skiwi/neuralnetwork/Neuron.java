package com.skiwi.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * @author Frank van Heeswijk
 */
public class Neuron {
    private final List<NeuronConnection> inputs = new ArrayList<>();
    private final List<NeuronConnection> outputs = new ArrayList<>();

    private double value = 0d;
    private double delta = 0d;

    public void setValue(double value) {
        this.value = value;
    }

    public double getValue() {
        return this.value;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getDelta() {
        return this.delta;
    }

    public void addInput(NeuronConnection neuronConnection) {
        this.inputs.add(neuronConnection);
    }

    public void addOutput(NeuronConnection neuronConnection) {
        this.outputs.add(neuronConnection);
    }

    void initializeWeights(Random random) {
        inputs.forEach(it -> it.setWeight(random.nextDouble() - 0.5d));
    }

    double getInputSum() {
        return inputs.stream().mapToDouble(NeuronConnection::getInValue).sum();
    }

    double getWeightedOutputDelta() {
        return outputs.stream().mapToDouble(NeuronConnection::getOutDelta).sum();
    }

    void forwardPropagate(DoubleUnaryOperator activationFunction) {
        value = activationFunction.applyAsDouble(getInputSum());
    }

    void calculateDeltaValue(DoubleUnaryOperator activationDerivativeFunction) {
        if (outputs.isEmpty()) {
            throw new IllegalStateException("This is an output neuron");
        }
        delta = activationDerivativeFunction.applyAsDouble(getInputSum()) * getWeightedOutputDelta();
    }

    void calculateDeltaValue(DoubleUnaryOperator activationDerivativeFunction, double error) {
        if (!outputs.isEmpty()) {
            throw new IllegalStateException("This is not an output neuron");
        }
        delta = activationDerivativeFunction.applyAsDouble(getInputSum()) * error;
    }

    void updateWeights(double learningRate) {
        inputs.forEach(it -> it.updateWeight(learningRate));
    }
}
