package com.skiwi.neuralnetwork

import java.util.function.DoubleUnaryOperator

import static com.skiwi.neuralnetwork.LayerType.HIDDEN
import static com.skiwi.neuralnetwork.LayerType.INPUT
import static com.skiwi.neuralnetwork.LayerType.OUTPUT

/**
 * @author Frank van Heeswijk
 */
class SimpleNeuralNetwork implements NeuralNetwork {
    List<NeuronLayer> layers

    SimpleNeuralNetwork(int inputNeuronCount, int outputNeuronCount, int... hiddenNeuronCount) {
        generateNetwork(inputNeuronCount, outputNeuronCount, hiddenNeuronCount)
    }

    private void generateNetwork(int inputNeuronCount, int outputNeuronCount, int... hiddenNeuronCount) {
        layers = []
        layers << NeuronLayer.create(INPUT, inputNeuronCount)
        hiddenNeuronCount.each { layers << NeuronLayer.create(HIDDEN, it) }
        layers << NeuronLayer.create(OUTPUT, outputNeuronCount)

        def biasNeuron = new Neuron(value: 1d)
        for (int i = 1; i < layers.size(); i++) {
            layers[i].addBiasNode(biasNeuron)
        }

        for (int i = 0; i < layers.size() - 1; i++) {
            NeuronLayer.connect(layers[i], layers[i + 1])
        }
    }

    @Override
    void learn(LearningData learningData, double learningRate, DoubleUnaryOperator activationFunction, DoubleUnaryOperator activationDerivativeFunction, int maxIterations) {
        def random = new Random()
        layers.each { it.initializeNeuronWeights(random) }
        int iterations = 0
        outer: while (true) {
            for (def data : learningData.data) {
                if (iterations++ == maxIterations) {
                    break outer
                }

                def queryVector = data.queryVector
                def targetVector = data.targetVector

                //forward propagation
                layers.first().queryVector = queryVector
                for (int i = 1; i < layers.size(); i++) {
                    layers[i].forwardPropagate(activationFunction)
                }

                //square error calculation
                def errorVector = layers.last().calculateErrorVector(targetVector)
                def squareError = Arrays.stream(errorVector).map({ Math.pow(it, 2) }).sum()

                //backward propagation
                layers.last().calculateDeltaValues(activationDerivativeFunction, targetVector)
                for (int i = layers.size() - 2; i >= 0; i--) {
                    layers[i].calculateDeltaValues(activationDerivativeFunction)
                }
                for (int i = 1; i < layers.size(); i++) {
                    layers[i].updateWeights(learningRate)
                }

                if (iterations % 100 == 0) {
                    println "Iteration $iterations/$maxIterations, Epoch = ${iterations / learningData.data.size()}, Square error = $squareError"
                }
            }
        }
    }

    @Override
    double[] query(double[] queryVector, DoubleUnaryOperator activationFunction) {
        layers.first().queryVector = queryVector
        for (int i = 1; i < layers.size(); i++) {
            layers[i].forwardPropagate(activationFunction)
        }

        layers.last().outputVector
    }
}
