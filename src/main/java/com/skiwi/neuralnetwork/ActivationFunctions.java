package com.skiwi.neuralnetwork;

import java.util.function.DoubleUnaryOperator;

/**
 * @author Frank van Heeswijk
 */
public class ActivationFunctions {
    public static final DoubleUnaryOperator SIGMOID_FUNCTION = it -> 1d / (1d + Math.exp(-it));
    public static final DoubleUnaryOperator SIGMOID_DERIVATIVE_FUNCTION = it -> SIGMOID_FUNCTION.applyAsDouble(it) * (1d - SIGMOID_FUNCTION.applyAsDouble(it));
}
