package com.skiwi.neuralnetwork

import spock.lang.Specification

/**
 * @author Frank van Heeswijk
 */
class SimpleNeuralNetworkTest extends Specification {
    void "test generate network"() {
        when: "create a simple neural network with 2 hidden layers"
        def neuralNetwork = new SimpleNeuralNetwork(100, 10, 40, 20)

        then: "network has to be generated correctly"
        neuralNetwork.layers.length == 4
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
            assert it.inputs.input.toSet() == firstNeuronSet
        }
    }
}
