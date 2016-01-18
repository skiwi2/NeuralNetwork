package com.skiwi.neuralnetwork;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author Frank van Heeswijk
 */
public class LearningData {
    private final List<LearningDatum> data;

    public LearningData(List<LearningDatum> data) {
        this.data = data;
    }

    public List<LearningDatum> getData() {
        return this.data;
    }

    public static <T> LearningData fromStream(Stream<T> stream, Function<T, LearningDatum> converter) {
        List<LearningDatum> data = stream.map(converter).collect(Collectors.toList());
        return new LearningData(data);
    }
}
