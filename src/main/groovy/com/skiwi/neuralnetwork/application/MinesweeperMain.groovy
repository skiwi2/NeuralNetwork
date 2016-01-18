package com.skiwi.neuralnetwork.application

import com.skiwi.neuralnetwork.LearningData
import com.skiwi.neuralnetwork.LearningDatum
import com.skiwi.neuralnetwork.SimpleNeuralNetwork

import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.util.stream.DoubleStream

import static com.skiwi.neuralnetwork.ActivationFunctions.SIGMOID_FUNCTION
import static com.skiwi.neuralnetwork.ActivationFunctions.SIGMOID_DERIVATIVE_FUNCTION

/**
 * @author Frank van Heeswijk
 */
class MinesweeperMain {
    void run() {
        def fileName = "challenge-flags-16x16.png"
        def url = getClass().getResource(fileName)
        def image = ImageIO.read(url)

        def data = [
            [getSubImageGrayScalePixels(image, 619, 617, 39, 39), [1d, 0d, 0d] as double[]],    //unclicked
            [getSubImageGrayScalePixels(image, 662, 617, 39, 39), [0d, 1d, 0d] as double[]],    //clicked
            [getSubImageGrayScalePixels(image, 790, 241, 39, 39), [0d, 1d, 0d] as double[]],    //clicked
            [getSubImageGrayScalePixels(image, 790, 197, 39, 39), [0d, 0d, 1d] as double[]],    //flag
            [getSubImageGrayScalePixels(image, 0, 0, 39, 39), [0d, 0d, 0d] as double[]]         //nothing
        ]
        def learningData = LearningData.fromStream(data.stream(), { new LearningDatum(it[0], it[1]) })
        def neuralNetwork = new SimpleNeuralNetwork(39 * 39, 3, 200)
        neuralNetwork.learn(learningData, 0.5d, SIGMOID_FUNCTION, SIGMOID_DERIVATIVE_FUNCTION, 2000)

        println ""
        println "Identifier: [UNCLICKED, CLICKED, FLAG]"
        println ""
        println "Test Unclicked: " + neuralNetwork.query(getSubImageGrayScalePixels(image, 619, 617, 39, 39), SIGMOID_FUNCTION)
        println "Test Clicked: " + neuralNetwork.query(getSubImageGrayScalePixels(image, 662, 617, 39, 39), SIGMOID_FUNCTION)
        println "Test Clicked: "+ neuralNetwork.query(getSubImageGrayScalePixels(image, 790, 241, 39, 39), SIGMOID_FUNCTION)
        println "Test Flag: " + neuralNetwork.query(getSubImageGrayScalePixels(image, 790, 197, 39, 39), SIGMOID_FUNCTION)
        println "Test Nothing: " + neuralNetwork.query(getSubImageGrayScalePixels(image, 0, 0, 39, 39), SIGMOID_FUNCTION)
        println ""
        println "Clicked 2: " + neuralNetwork.query(getSubImageGrayScalePixels(image, 834, 284, 39, 39), SIGMOID_FUNCTION)
        println "Clicked 2: " + neuralNetwork.query(getSubImageGrayScalePixels(image, 1178, 370, 39, 39), SIGMOID_FUNCTION)
        println "Clicked 4: "+ neuralNetwork.query(getSubImageGrayScalePixels(image, 1092, 327, 39, 39), SIGMOID_FUNCTION)
        println "Flag: " + neuralNetwork.query(getSubImageGrayScalePixels(image, 1049, 325, 39, 39), SIGMOID_FUNCTION)
        println "Middle Junk: " + neuralNetwork.query(getSubImageGrayScalePixels(image, 1000, 500, 39, 39), SIGMOID_FUNCTION)
    }

    static double[] getSubImageGrayScalePixels(BufferedImage image, int x, int y, int dx, int dy) {
        def rgbPixels = image.getRGB(x, y, dx, dy, null, 0, dx)
        Arrays.stream(rgbPixels).boxed().flatMapToDouble({
            DoubleStream.of(getGray(
                (it >> 16) & 0xFF,
                (it >> 8) & 0xFF,
                it & 0xFF
            ))
        }).toArray()
    }

    static double getGray(int red, int green, int blue) {
        (0.2989d * red + 0.5870d * green + 0.1140d * blue) / 255d
    }

    public static void main(String[] args) {
        new MinesweeperMain().run()
    }
}
