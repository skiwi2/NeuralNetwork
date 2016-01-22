package com.skiwi.neuralnetwork.application;

import com.skiwi.neuralnetwork.LearningData;
import com.skiwi.neuralnetwork.LearningDatum;
import com.skiwi.neuralnetwork.NeuralNetwork;
import com.skiwi.neuralnetwork.SimpleNeuralNetwork;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static com.skiwi.neuralnetwork.ActivationFunctions.SIGMOID_DERIVATIVE_FUNCTION;
import static com.skiwi.neuralnetwork.ActivationFunctions.SIGMOID_FUNCTION;

/**
 * @author Frank van Heeswijk
 */
public class MNISTImageRecognition {
    private static final Path MNIST_DIRECTORY = Paths.get(System.getProperty("user.home")).resolve("NN-data/MNIST");

    void run() throws IOException {
        InputStream trainingImageStream = Files.newInputStream(MNIST_DIRECTORY.resolve("train-images.idx3-ubyte"));
        InputStream trainingLabelStream = Files.newInputStream(MNIST_DIRECTORY.resolve("train-labels.idx1-ubyte"));
        LearningData trainingData = createTrainingData(trainingImageStream, trainingLabelStream);
        trainingImageStream.close();
        trainingLabelStream.close();

        InputStream testImageStream = Files.newInputStream(MNIST_DIRECTORY.resolve("t10k-images.idx3-ubyte"));
        InputStream testLabelStream = Files.newInputStream(MNIST_DIRECTORY.resolve("t10k-labels.idx1-ubyte"));
        LearningData testData = createTestData(testImageStream, testLabelStream);
        testImageStream.close();
        testLabelStream.close();

        NeuralNetwork neuralNetwork = new SimpleNeuralNetwork(784, 10, 30);
        neuralNetwork.learn(trainingData, 0.3d, SIGMOID_FUNCTION, SIGMOID_DERIVATIVE_FUNCTION, 60000 * 30);

        System.out.println();

        int tests = testData.getData().size();
        int correct = 0;

        for (int i = 0; i < tests; i++) {
            LearningDatum testDatum = testData.getData().get(i);
            double[] queryVector = testDatum.getQueryVector();
            double[] targetVector = testDatum.getTargetVector();

            double[] outputVector = neuralNetwork.query(queryVector, SIGMOID_FUNCTION);
            int maxLabel = 0;
            double max = outputVector[0];
            for (int j = 1; j < 10; j++) {
                if (outputVector[j] > max) {
                    max = outputVector[j];
                    maxLabel = j;
                }
            }

            if (maxLabel == targetVector.length) {
                correct++;
            }
        }

        System.out.println("Correct: " + correct + "/" + tests + " (" + (correct * 1d / tests * 100d) + "%)");
    }

    LearningData createTrainingData(InputStream imageStream, InputStream labelStream) throws IOException {
        byte[] headerData = new byte[4];

        imageStream.read(headerData);
        int imageMagicNumber = ByteBuffer.wrap(headerData).getInt();
        imageStream.read(headerData);
        int imageCount = ByteBuffer.wrap(headerData).getInt();
        imageStream.read(headerData);
        int imageRowCount = ByteBuffer.wrap(headerData).getInt();
        imageStream.read(headerData);
        int imageColumnCount = ByteBuffer.wrap(headerData).getInt();
        int imagePixels = imageRowCount * imageColumnCount;

        labelStream.read(headerData);
        int labelMagicNumber = ByteBuffer.wrap(headerData).getInt();
        labelStream.read(headerData);
        int labelCount = ByteBuffer.wrap(headerData).getInt();

        if (imageMagicNumber != 2051) {
            throw new IllegalStateException("Invalid image magic number: " + imageMagicNumber);
        }
        if (labelMagicNumber != 2049) {
            throw new IllegalStateException("Invalid label magic number: " + labelMagicNumber);
        }
        if (imageCount != labelCount) {
            throw new IllegalStateException("Image count does not match label count: " + imageCount + " != " + labelCount);
        }

        byte[] pixelData = new byte[imagePixels];
        byte[] labelData = new byte[1];

        List<LearningDatum> learningData = new ArrayList<>();
        for (int i = 0; i < imageCount; i++) {
            imageStream.read(pixelData);
            labelStream.read(labelData);

            double[] queryVector = new double[imagePixels];
            double[] targetVector = new double[10];

            for (int j = 0; j < imagePixels; j++) {
                queryVector[j] = (Byte.toUnsignedInt(pixelData[j]) / 255d);
            }
            targetVector[labelData[0]] = 1d;

            learningData.add(new LearningDatum(queryVector, targetVector));
        }

        return LearningData.fromStream(learningData.stream(), i -> i);
    }

    LearningData createTestData(InputStream imageStream, InputStream labelStream) throws IOException {
        byte[] headerData = new byte[4];

        imageStream.read(headerData);
        int imageMagicNumber = ByteBuffer.wrap(headerData).getInt();
        imageStream.read(headerData);
        int imageCount = ByteBuffer.wrap(headerData).getInt();
        imageStream.read(headerData);
        int imageRowCount = ByteBuffer.wrap(headerData).getInt();
        imageStream.read(headerData);
        int imageColumnCount = ByteBuffer.wrap(headerData).getInt();
        int imagePixels = imageRowCount * imageColumnCount;

        labelStream.read(headerData);
        int labelMagicNumber = ByteBuffer.wrap(headerData).getInt();
        labelStream.read(headerData);
        int labelCount = ByteBuffer.wrap(headerData).getInt();

        if (imageMagicNumber != 2051) {
            throw new IllegalStateException("Invalid image magic number: " + imageMagicNumber);
        }
        if (labelMagicNumber != 2049) {
            throw new IllegalStateException("Invalid label magic number: " + labelMagicNumber);
        }
        if (imageCount != labelCount) {
            throw new IllegalStateException("Image count does not match label count: " + imageCount + " != " + labelCount);
        }

        byte[] pixelData = new byte[imagePixels];
        byte[] labelData = new byte[1];

        List<LearningDatum> learningData = new ArrayList<>();
        for (int i = 0; i < imageCount; i++) {
            imageStream.read(pixelData);
            labelStream.read(labelData);

            double[] queryVector = new double[imagePixels];
            double[] targetVector = new double[labelData[0]];   //ugly hack

            for (int j = 0; j < imagePixels; j++) {
                queryVector[j] = (Byte.toUnsignedInt(pixelData[j]) / 255d);
            }

            learningData.add(new LearningDatum(queryVector, targetVector));
        }

        return LearningData.fromStream(learningData.stream(), i -> i);
    }

    public static void main(String[] args) throws IOException {
        new MNISTImageRecognition().run();
    }
}
