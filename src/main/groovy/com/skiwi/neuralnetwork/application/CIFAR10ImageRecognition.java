package com.skiwi.neuralnetwork.application;

import com.skiwi.neuralnetwork.LearningData;
import com.skiwi.neuralnetwork.LearningDatum;
import com.skiwi.neuralnetwork.NeuralNetwork;
import com.skiwi.neuralnetwork.SimpleNeuralNetwork;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
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
public class CIFAR10ImageRecognition {
    private static final Path CIFAR10_DIRECTORY = Paths.get(System.getProperty("user.home")).resolve("NN-data/CIFAR10");

    void run() throws IOException {
        String[] classes = new String[10];
        try (BufferedReader metaReader = Files.newBufferedReader(CIFAR10_DIRECTORY.resolve("batches.meta.txt"))) {
            for (int i = 0; i < 10; i++) {
                classes[i] = metaReader.readLine();
            }
        }

        List<LearningDatum> data = new ArrayList<>();
        for (int batch = 1; batch <= 5; batch++) {
            try (InputStream batchStream = Files.newInputStream(CIFAR10_DIRECTORY.resolve("data_batch_" + batch + ".bin"))) {
                byte[] bytes = new byte[3073];
                for (int i = 0; i < 10000; i++) {
                    batchStream.read(bytes);
                    data.add(createTrainingDataFromByteArray(bytes));
                }
            }
        }

        List<LearningDatum> testData = new ArrayList<>();
        try (InputStream batchStream = Files.newInputStream(CIFAR10_DIRECTORY.resolve("test_batch.bin"))) {
            byte[] bytes = new byte[3073];
            for (int i = 0; i < 10000; i++) {
                batchStream.read(bytes);
                testData.add(createTestDataFromByteArray(bytes));
            }
        }

        LearningData learningData = LearningData.fromStream(data.stream(), i -> i);
        NeuralNetwork neuralNetwork = new SimpleNeuralNetwork(3072, 10, 10);
        neuralNetwork.learn(learningData, 0.1d, SIGMOID_FUNCTION, SIGMOID_DERIVATIVE_FUNCTION, 1000000);


        int[] classes50Correct = new int[10];
        int[] classes60Correct = new int[10];
        int[] classes70Correct = new int[10];
        int[] classes80Correct = new int[10];
        int[] classes90Correct = new int[10];
        int j = 0;
        for (LearningDatum datum : testData) {
            int classIdentifier = datum.getTargetVector().length;
            double[] queryVector = datum.getQueryVector();
            //checking 90% confidence
            double[] outputVector = neuralNetwork.query(queryVector, SIGMOID_FUNCTION);
            if (outputVector[classIdentifier] >= 0.90d) {
                classes90Correct[classIdentifier]++;
            }
            else if (outputVector[classIdentifier] >= 0.80d) {
                classes80Correct[classIdentifier]++;
            }
            else if (outputVector[classIdentifier] >= 0.70d) {
                classes70Correct[classIdentifier]++;
            }
            else if (outputVector[classIdentifier] >= 0.60d) {
                classes60Correct[classIdentifier]++;
            }
            else if (outputVector[classIdentifier] >= 0.50d) {
                classes50Correct[classIdentifier]++;
            }
            if (++j % 100 == 0) {
                System.out.println("Testing... " + j + "/10000");
            }
        }

        System.out.println("");
        System.out.println("===Test data correctness (50% confidence)===");
        for (int i = 0; i < 10; i++) {
            System.out.println(classes[i] + " " + ((classes50Correct[i] / 1000) * 100d) + "%");
        }

        System.out.println("");
        System.out.println("===Test data correctness (60% confidence)===");
        for (int i = 0; i < 10; i++) {
            System.out.println(classes[i] + " " + ((classes60Correct[i] / 1000) * 100d) + "%");
        }

        System.out.println("");
        System.out.println("===Test data correctness (70% confidence)===");
        for (int i = 0; i < 10; i++) {
            System.out.println(classes[i] + " " + ((classes70Correct[i] / 1000) * 100d) + "%");
        }

        System.out.println("");
        System.out.println("===Test data correctness (80% confidence)===");
        for (int i = 0; i < 10; i++) {
            System.out.println(classes[i] + " " + ((classes80Correct[i] / 1000) * 100d) + "%");
        }

        System.out.println("");
        System.out.println("===Test data correctness (90% confidence)===");
        for (int i = 0; i < 10; i++) {
            System.out.println(classes[i] + " " + ((classes90Correct[i] / 1000) * 100d) + "%");
        }
    }

    LearningDatum createTrainingDataFromByteArray(byte[] bytes) {
        int classIdentifier = Byte.toUnsignedInt(bytes[0]);
        double[] queryVector = new double[3072];
        for (int i = 1; i < bytes.length; i++) {
            queryVector[i - 1] = (Byte.toUnsignedInt(bytes[i]) / 255d);
        }
        double[] targetVector = new double[10];
        targetVector[classIdentifier] = 1d;
        return new LearningDatum(queryVector, targetVector);
    }

    LearningDatum createTestDataFromByteArray(byte[] bytes) {
        int classIdentifier = Byte.toUnsignedInt(bytes[0]);
        double[] queryVector = new double[3072];
        for (int i = 1; i < bytes.length; i++) {
            queryVector[i - 1] = (Byte.toUnsignedInt(bytes[i]) / 255d);
        }
        //TODO fix ugly hack
        double[] targetVector = new double[classIdentifier];
        return new LearningDatum(queryVector, targetVector);
    }

    public static void main(String[] args) throws IOException {
        new CIFAR10ImageRecognition().run();
    }
}
