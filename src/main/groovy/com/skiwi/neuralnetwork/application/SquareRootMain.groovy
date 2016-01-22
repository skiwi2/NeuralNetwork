package com.skiwi.neuralnetwork.application

import com.skiwi.neuralnetwork.LearningData
import com.skiwi.neuralnetwork.LearningDatum
import com.skiwi.neuralnetwork.SimpleNeuralNetwork

import static com.skiwi.neuralnetwork.ActivationFunctions.SIGMOID_DERIVATIVE_FUNCTION
import static com.skiwi.neuralnetwork.ActivationFunctions.SIGMOID_FUNCTION

/**
 * @author Frank van Heeswijk
 */
class SquareRootMain {
    void run() {
        def data = []
        for (int i = 0; i <= 1000; i++) {
            data << [[i * i / 1000000d] as double[], [i / 1000000d] as double[]]
        }
        def learningData = LearningData.fromStream(data.stream(), { new LearningDatum(it[0], it[1]) })
        def neuralNetwork = new SimpleNeuralNetwork(1, 1, 10)
        neuralNetwork.learn(learningData, 0.0000005d, SIGMOID_FUNCTION, SIGMOID_DERIVATIVE_FUNCTION, 1000000000)

        for (int i = 0; i <= 100; i++) {
            int square = i * i;
            println "SQRT $square: " + (neuralNetwork.query([square / 1000000d] as double[], SIGMOID_FUNCTION))[0] * 1000000d
        }

//        println "SQRT 25: " + (neuralNetwork.query([25.0d / 1000000d] as double[], SIGMOID_FUNCTION)[0] * 1000000d)
//        println "SQRT 100: " + (neuralNetwork.query([100.0d / 1000000d] as double[], SIGMOID_FUNCTION)[0] * 1000000d)
//        println "SQRT 1000: " + (neuralNetwork.query([1000.0d / 1000000d] as double[], SIGMOID_FUNCTION)[0] * 1000000d)
//        println "SQRT 1521: " + (neuralNetwork.query([1521.0d / 1000000d] as double[], SIGMOID_FUNCTION)[0] * 1000000d)
//        println ""
//        println "SQRT 4000000: " + (neuralNetwork.query([4000000.0d / 1000000d] as double[], SIGMOID_FUNCTION)[0] * 1000000d)
//        println "SQRT 550042: " + (neuralNetwork.query([550042.0d / 1000000d] as double[], SIGMOID_FUNCTION)[0] * 1000000d)
//        println "SQRT 8: " + (neuralNetwork.query([8.0d / 1000000d] as double[], SIGMOID_FUNCTION)[0] * 1000000d)
//        println "SQRT 42: " + (neuralNetwork.query([42.0d / 1000000d] as double[], SIGMOID_FUNCTION)[0] * 1000000d)
    }

    public static void main(String[] args) {
        new SquareRootMain().run()
    }
}
