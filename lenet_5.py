# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
    
def LeNet(features, labels, mode):
    # Input Layer
    # The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.
    # [batch_size, image_height, image_width, channels] 
    # Note that we've indicated -1 for batch size, 
    # which specifies that this dimension should be
    # dynamically computed based on the number of 
    # input values in features["x"] 
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 1])

    # Architecture

    # Layer 1: Convolutional. The output shape should be 28x28x6.
    # number of filters = depth of output = 6
    # (W - F + 2P)/ S + 1
    # 28 = (32 - 5 + 2P)/S + 1
    # choose size of filters to be 5 by 5 (F = 5)
    # Activation. Your choice of activation function.
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=6,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)
    
    # Pooling. The output shape should be 14x14x6.
    # (W - F)/S + 1 = 14
    # (28 - 2)/2 + 1 
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Layer 2: Convolutional. The output shape should be 10x10x16.
    # 16 filters 
    # 5 by 5 again
    # (14 - 5)/ 1 + 1 = 10
    # Activation. Your choice of activation function.
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=16,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)

    # Pooling. The output shape should be 5x5x16.
    # (W - F)/S + 1 = 5
    # (10 - 2)/2 + 1 = 5
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.
    from tensorflow.contrib.layers import flatten
    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool2)
    # or:  pool2_flat = tf.reshape(pool2, [-1, 400])

    # Layer 3: Fully Connected. This should have 120 outputs.
    # Activation. Your choice of activation function.
    # Fully Connected. Input = 400. Output = 120.
    fc1dense = tf.layers.dense(inputs=fc1, units=120, activation=tf.nn.relu)

    # Layer 4: Fully Connected. This should have 84 outputs.
    # Activation. Your choice of activation function.
    fc2dense = tf.layers.dense(inputs=fc1dense, units=84, activation=tf.nn.relu)

    # Layer 5: Fully Connected (Logits). This should have 10 outputs.
    logits = tf.layers.dense(inputs=fc2dense, units=10)

      # predictions dictionary
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)

      # The input argument specifies the tensor from which to 
      # extract maximum values—here logits. The axis argument 
      # specifies the axis of the input tensor along which to 
      # find the greatest value. Here, we want to find the 
      # largest value along the dimension with index of 1, 
      # which corresponds to our predictions 
      # (recall that our logits tensor has shape 
      # [batch_size, 10]).

      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes) (scalar)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    from tensorflow.examples.tutorials.mnist import input_data

    # Load the MNIST data, which comes pre-loaded with TensorFlow.
    # mnist = input_data.read_data_sets("MNIST_data/", reshape = False)
    # x_train, y_train = mnist.train.images, mnist.train.labels
    # x_validation, y_validation = mnist.validation.images, mnist.validation.labels
    # x_test, y_test = mnist.test.images, mnist.test.labels
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    #mnist = input_data.read_data_sets("MNIST_data/", reshape = False)
    x_train = mnist.train.images # Returns np.array
    x_train = x_train.reshape([-1, 28, 28, 1])
    y_train = np.asarray(mnist.train.labels, dtype=np.int32)
    # y_train = y_train.reshape([-1, 28, 28, 1])
    x_test = mnist.test.images # Returns np.array
    x_test = x_test.reshape([-1, 28, 28, 1])
    y_test = np.asarray(mnist.test.labels, dtype=np.int32)
    # y_train = y_train.reshape([-1, 28, 28, 1])

    # check if data size is proper
    # assert(len(x_train) == len(y_train))
    # assert(len(x_validation) == len(y_validation))
    # assert(len(x_test) == len(y_test))

    print()
    print("Image Shape: {}".format(x_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(x_train)))
    # print("Validation Set: {} samples".format(len(x_validation)))
    print("Test Set:       {} samples".format(len(x_test)))

    # The MNIST data that TensorFlow pre-loads comes as 
    # 28x28x1 images.
    # However, the LeNet architecture only accepts 32x32xC 
    # images, where C is the number of color channels.
    # In order to reformat the MNIST data into a shape that 
    # LeNet will accept, we pad the data with two rows of zeros 
    # on the top and bottom, and two columns of zeros on the 
    # left and right (28+2+2 = 32).

    # Pad images with 0s
    x_train      = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    # x_validation = np.pad(x_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    x_test       = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        
    print("Updated Image Shape: {}".format(x_train[0].shape))

    # View a sample from the dataset.
    # import random
    # import numpy as np
    # import matplotlib.pyplot as plt
    # %matplotlib inline

    # index = random.randint(0, len(x_train))
    # image = x_train[index].squeeze()

    # plt.figure(figsize=(1,1))
    # plt.imshow(image, cmap="gray")
    # print(y_train[index])

    # Preprocess Data
    # Shuffle the training data.
    # x_train, y_train = shuffle(x_train, y_train)

    mnist_classifier = tf.estimator.Estimator(
    model_fn=LeNet, model_dir="/tmp/mnist_lenet_model")

    # after every 50 steps of training.
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
        
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()


