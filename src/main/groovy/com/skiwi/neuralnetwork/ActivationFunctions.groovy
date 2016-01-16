package com.skiwi.neuralnetwork

import java.util.function.DoubleUnaryOperator

/**
 * @author Frank van Heeswijk
 */
class ActivationFunctions {
    static final DoubleUnaryOperator SIGMOID_FUNCTION = { 1d / (1 + Math.exp(-it)) }
    static final DoubleUnaryOperator SIGMOID_DERIVATIVE_FUNCTION = { SIGMOID_FUNCTION.applyAsDouble(it) * (1d - SIGMOID_FUNCTION.applyAsDouble(it)) }
}
