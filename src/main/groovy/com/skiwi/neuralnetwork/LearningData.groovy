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

    static <T> LearningData fromStream(Stream<T> stream, Function<T, LearningDatum> converter) {
        def data = stream.collect { converter.apply(it) }
        return new LearningData(data)
    }
}
