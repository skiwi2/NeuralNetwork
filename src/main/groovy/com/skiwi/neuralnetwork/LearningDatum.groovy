package com.skiwi.neuralnetwork

/**
 * @author Frank van Heeswijk
 */
class LearningDatum {
    double[] queryVector
    double[] targetVector

    LearningDatum(double[] queryVector, double[] targetVector) {
        this.queryVector = queryVector
        this.targetVector = targetVector
    }
}
