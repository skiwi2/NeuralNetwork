# NeuralNetwork
A collection of neural networks.

Input data
==========
The data is read from `%USER_HOME%/NN-data/`, a detailed list of the required data is available below.

MNIST
-----
The MNIST data can be obtained from http://yann.lecun.com/exdb/mnist/, the four files can be found at the top of the file.

Required files:

 - train-images-idx3-ubyte
 - train-labels-idx1-ubyte
 - t10k-images-idx3-ubyte
 - t10k-labels-idx1-ubyte
 
 These files should be placed under `%USER_HOME%/NN-data/MNIST/`.
 
 **Note**: It is expected that the files are uncompressed.
 
 CIFAR10
 -----
 The CIFAR10 data can be obtained from http://www.cs.utoronto.ca/~kriz/cifar.html, under downloads the binary version should be picked.
 
 Required files:
 
  - batches.meta.txt
  - data_batch_1.bin
  - data_batch_2.bin
  - data_batch_3.bin
  - data_batch_4.bin
  - data_batch_5.bin
  - test_batch.bin
  
  These files should be placed under `%USER_HOME%/NN-data/CIFAR10/`.
  
  **Note**: It is expected that the files are uncompressed and unzipped.