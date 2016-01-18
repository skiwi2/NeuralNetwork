package com.skiwi.neuralnetwork;

/**
 * @author Frank van Heeswijk
 */
public class LearningDatum {
    private final double[] queryVector;
    private final double[] targetVector;

    public LearningDatum(double[] queryVector, double[] targetVector) {
        this.queryVector = queryVector;
        this.targetVector = targetVector;
    }

    public double[] getQueryVector() {
        return this.queryVector;
    }

    public double[] getTargetVector() {
        return this.targetVector;
    }
}
