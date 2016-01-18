package com.skiwi.neuralnetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

import static com.skiwi.neuralnetwork.LayerType.HIDDEN;
import static com.skiwi.neuralnetwork.LayerType.INPUT;
import static com.skiwi.neuralnetwork.LayerType.OUTPUT;

/**
 * @author Frank van Heeswijk
 */
public class SimpleNeuralNetwork implements NeuralNetwork {
    private List<NeuronLayer> layers;

    public SimpleNeuralNetwork(int inputNeuronCount, int outputNeuronCount, int... hiddenNeuronCount) {
        generateNetwork(inputNeuronCount, outputNeuronCount, hiddenNeuronCount);
    }

    private void generateNetwork(int inputNeuronCount, int outputNeuronCount, int... hiddenNeuronCount) {
        layers = new ArrayList<>();
        layers.add(NeuronLayer.create(INPUT, inputNeuronCount));
        for (int layerNeuronCount : hiddenNeuronCount) {
            layers.add(NeuronLayer.create(HIDDEN, layerNeuronCount));
        }
        layers.add(NeuronLayer.create(OUTPUT, outputNeuronCount));

        Neuron biasNeuron = new Neuron();
        biasNeuron.setValue(1d);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).addBiasNode(biasNeuron);
        }

        for (int i = 0; i < layers.size() - 1; i++) {
            NeuronLayer.connect(layers.get(i), layers.get(i + 1));
        }
    }

    @Override
    public void learn(LearningData learningData, double learningRate, DoubleUnaryOperator activationFunction, DoubleUnaryOperator activationDerivativeFunction, int maxIterations) {
        Random random = new Random();
        layers.forEach(it -> it.initializeNeuronWeights(random));
        int iterations = 0;
        outer: while (true) {
            for (LearningDatum data : learningData.getData()) {
                if (iterations++ == maxIterations) {
                    break outer;
                }

                double[] queryVector = data.getQueryVector();
                double[] targetVector = data.getTargetVector();

                //forward propagation
                layers.get(0).setQueryVector(queryVector);
                for (int i = 1; i < layers.size(); i++) {
                    layers.get(i).forwardPropagate(activationFunction);
                }

                //square error calculation
                double[] errorVector = layers.get(layers.size() - 1).calculateErrorVector(targetVector);
                double squareError = Arrays.stream(errorVector).map(it -> Math.pow(it, 2d)).sum();

                //backward propagation
                layers.get(layers.size() - 1).calculateDeltaValues(activationDerivativeFunction, targetVector);
                for (int i = layers.size() - 2; i >= 0; i--) {
                    layers.get(i).calculateDeltaValues(activationDerivativeFunction);
                }
                for (int i = 1; i < layers.size(); i++) {
                    layers.get(i).updateWeights(learningRate);
                }

                if (iterations % 100 == 0) {
                    System.out.println("Iteration " + iterations + "/" + maxIterations + ", Epoch = " + (iterations / learningData.getData().size()) + ", Square error = " + squareError);
                }
            }
        }
    }

    @Override
    public double[] query(double[] queryVector, DoubleUnaryOperator activationFunction) {
        layers.get(0).setQueryVector(queryVector);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).forwardPropagate(activationFunction);
        }

        return layers.get(layers.size() - 1).getOutputVector();
    }
}