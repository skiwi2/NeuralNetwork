package com.skiwi.neuralnetwork

import spock.lang.Specification

import static spock.util.matcher.HamcrestMatchers.closeTo
import static spock.util.matcher.HamcrestSupport.that

/**
 * @author Frank van Heeswijk
 */
class LearningDataTest extends Specification {
    void "test from stream"() {
        when: "learning data gets converted from stream"
        def list = ["1 0 0 | 0 1 0", "1 1 0 | 0 1 1", "1 1 1 | 1 0 0"]
        def converter = { String str ->
            def parts = str.split("\\|")
            def partToDoubleArray = { String it -> it.trim().split(" ").toList().stream().mapToDouble({ Double.parseDouble(it) }).toArray() }
            def queryVector = partToDoubleArray(parts[0])
            def targetVector = partToDoubleArray(parts[1])
            new LearningDatum(queryVector, targetVector)
        }
        def learningData = LearningData.fromStream(list.stream(), converter)

        then: "learning data should be correct"
        doubleArrayIsCloseTo(learningData.data[0].queryVector, [1d, 0d, 0d] as double[], 0.01d)
        doubleArrayIsCloseTo(learningData.data[0].targetVector, [0d, 1d, 0d] as double[], 0.01d)
        doubleArrayIsCloseTo(learningData.data[1].queryVector, [1d, 1d, 0d] as double[], 0.01d)
        doubleArrayIsCloseTo(learningData.data[1].targetVector, [0d, 1d, 1d] as double[], 0.01d)
        doubleArrayIsCloseTo(learningData.data[2].queryVector, [1d, 1d, 1d] as double[], 0.01d)
        doubleArrayIsCloseTo(learningData.data[2].targetVector, [1d, 0d, 0d] as double[], 0.01d)
    }

    def doubleArrayIsCloseTo(double[] first, double[] second, double error) {
        assert first.length == second.length
        first.eachWithIndex{ d, i ->
            assert that(d, closeTo(second[i], error))
        }
    }
}
