package com.skiwi.neuralnetwork

import java.util.function.Function
import java.util.stream.Stream

/**
 * @author Frank van Heeswijk
 */
class LearningData {
    List<LearningDatum> data

    LearningData(List<LearningDatum> data) {
        this.data = data
    }

    static LearningData fromStream(Stream<String> stream, Function<String, LearningDatum> converter) {
        def data = stream.collect { converter.apply(it) }
        return new LearningData(data)
    }
}
