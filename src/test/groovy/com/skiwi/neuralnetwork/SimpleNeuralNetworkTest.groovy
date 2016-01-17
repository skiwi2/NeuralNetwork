package com.skiwi.neuralnetwork

import spock.lang.Specification

import static com.skiwi.neuralnetwork.ActivationFunctions.SIGMOID_DERIVATIVE_FUNCTION
import static com.skiwi.neuralnetwork.ActivationFunctions.SIGMOID_FUNCTION
import static spock.util.matcher.HamcrestMatchers.closeTo
import static spock.util.matcher.HamcrestSupport.that

/**
 * @author Frank van Heeswijk
 */
class SimpleNeuralNetworkTest extends Specification {
    void "test generate network"() {
        when: "create a simple neural network with 2 hidden layers"
        def neuralNetwork = new SimpleNeuralNetwork(100, 10, 40, 20)

        then: "network has to be generated correctly"
        neuralNetwork.layers.size() == 4
        neuralNetwork.layers[0].neurons.length == 100
        neuralNetwork.layers[1].neurons.length == 40
        neuralNetwork.layers[2].neurons.length == 20
        neuralNetwork.layers[3].neurons.length == 10
        neuralNetwork.layers[0].nextLayer == neuralNetwork.layers[1]
        neuralNetwork.layers[1].nextLayer == neuralNetwork.layers[2]
        neuralNetwork.layers[2].nextLayer == neuralNetwork.layers[3]
        neuralNetwork.layers[3].nextLayer == null
        neuralNetwork.layers[0].previousLayer == null
        neuralNetwork.layers[1].previousLayer == neuralNetwork.layers[0]
        neuralNetwork.layers[2].previousLayer == neuralNetwork.layers[1]
        neuralNetwork.layers[3].previousLayer == neuralNetwork.layers[2]
        nodesInLayerHaveMatchingInAndOut(neuralNetwork.layers[0], neuralNetwork.layers[1])
        nodesInLayerHaveMatchingInAndOut(neuralNetwork.layers[1], neuralNetwork.layers[2])
        nodesInLayerHaveMatchingInAndOut(neuralNetwork.layers[2], neuralNetwork.layers[3])
    }

    def nodesInLayerHaveMatchingInAndOut(NeuronLayer first, NeuronLayer second) {
        def firstNeuronSet = first.neurons.toList().toSet()
        def secondNeuronSet = second.neurons.toList().toSet()
        first.neurons.each { it ->
            assert it.outputs.output.toSet() == secondNeuronSet
        }
        second.neurons.each { it ->
            //ignore bias node
            assert it.inputs.input.findAll { it.value != 1.0d }.toSet() == firstNeuronSet
        }
    }

    void "test AND-function without hidden nodes"() {
        when: "neural network is trained for the AND-function"
        def neuralNetwork = new SimpleNeuralNetwork(2, 1)
        def data = [
            [[0d, 0d] as double[], [0d] as double[]],
            [[0d, 1d] as double[], [0d] as double[]],
            [[1d, 0d] as double[], [0d] as double[]],
            [[1d, 1d] as double[], [1d] as double[]]
        ]
        def learningData = LearningData.fromStream(data.stream(), { new LearningDatum(it[0], it[1]) })
        neuralNetwork.learn(learningData, 0.7d, SIGMOID_FUNCTION, SIGMOID_DERIVATIVE_FUNCTION, 100000)

        then: "outputs should be correct"
        doubleArrayIsCloseTo(neuralNetwork.query([0d, 0d] as double[], SIGMOID_FUNCTION), [0d] as double[], 0.01d)
        doubleArrayIsCloseTo(neuralNetwork.query([0d, 1d] as double[], SIGMOID_FUNCTION), [0d] as double[], 0.01d)
        doubleArrayIsCloseTo(neuralNetwork.query([1d, 0d] as double[], SIGMOID_FUNCTION), [0d] as double[], 0.01d)
        doubleArrayIsCloseTo(neuralNetwork.query([1d, 1d] as double[], SIGMOID_FUNCTION), [1d] as double[], 0.01d)
    }

    void "test XOR-function with hidden nodes"() {
        when: "neural network is trained for the XOR-function"
        def neuralNetwork = new SimpleNeuralNetwork(2, 1, 2)
        def data = [
            [[0d, 0d] as double[], [0d] as double[]],
            [[0d, 1d] as double[], [1d] as double[]],
            [[1d, 0d] as double[], [1d] as double[]],
            [[1d, 1d] as double[], [0d] as double[]]
        ]
        def learningData = LearningData.fromStream(data.stream(), { new LearningDatum(it[0], it[1]) })
        neuralNetwork.learn(learningData, 0.7d, SIGMOID_FUNCTION, SIGMOID_DERIVATIVE_FUNCTION, 100000)

        then: "outputs should be correct"
        doubleArrayIsCloseTo(neuralNetwork.query([0d, 0d] as double[], SIGMOID_FUNCTION), [0d] as double[], 0.01d)
        doubleArrayIsCloseTo(neuralNetwork.query([0d, 1d] as double[], SIGMOID_FUNCTION), [1d] as double[], 0.01d)
        doubleArrayIsCloseTo(neuralNetwork.query([1d, 0d] as double[], SIGMOID_FUNCTION), [1d] as double[], 0.01d)
        doubleArrayIsCloseTo(neuralNetwork.query([1d, 1d] as double[], SIGMOID_FUNCTION), [0d] as double[], 0.01d)
    }


    def doubleArrayIsCloseTo(double[] first, double[] second, double error) {
        assert first.length == second.length
        first.eachWithIndex{ d, i ->
            assert that(d, closeTo(second[i], error))
        }
    }
}
