package com.skiwi.neuralnetwork.application

import com.skiwi.neuralnetwork.LearningData
import com.skiwi.neuralnetwork.LearningDatum
import com.skiwi.neuralnetwork.SimpleNeuralNetwork

import static com.skiwi.neuralnetwork.ActivationFunctions.SIGMOID_DERIVATIVE_FUNCTION
import static com.skiwi.neuralnetwork.ActivationFunctions.SIGMOID_FUNCTION

/**
 * @author Frank van Heeswijk
 */
class SquareMain {
    void run() {
        def data = []
        def random = new Random()
//        1000.times {
//            def randomInt = random.nextInt(1000)
//            data << [[randomInt / 1000000d] as double[], [randomInt * randomInt / 1000000d] as double[]]
//        }
        def MAX = 10
        def RANGE_MAX = MAX * MAX as double
        for (int i = 0; i <= MAX; i++) {
            data << [[i / RANGE_MAX] as double[], [i * i / RANGE_MAX] as double[]]
        }
        def learningData = LearningData.fromStream(data.stream(), { new LearningDatum(it[0], it[1]) })
        def neuralNetwork = new SimpleNeuralNetwork(1, 1, 20)
        neuralNetwork.learn(learningData, 0.1d, SIGMOID_FUNCTION, SIGMOID_DERIVATIVE_FUNCTION, 10000)

        for (int i = 0; i <= 10; i++) {
            println "SQRUARE $i: " + (neuralNetwork.query([i / RANGE_MAX] as double[], SIGMOID_FUNCTION))[0] * RANGE_MAX
        }
    }

    public static void main(String[] args) {
        new SquareMain().run()
    }
}
