package com.skiwi.neuralnetwork;

/**
 * @author Frank van Heeswijk
 */
public class NeuronConnection {
    private final Neuron input;
    private final Neuron output;

    private double weight = 0d;

    public NeuronConnection(Neuron input, Neuron output) {
        this.input = input;
        this.output = output;
    }

    public Neuron getInput() {
        return input;
    }

    public Neuron getOutput() {
        return output;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getWeight() {
        return weight;
    }

    double getInValue() {
        return weight * input.getValue();
    }

    double getOutValue() {
        return weight * output.getValue();
    }

    double getOutDelta() {
        return weight * output.getDelta();
    }

    void updateWeight(double learningRate) {
        weight += learningRate * input.getValue() * output.getDelta();
    }
}
