# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # [batch_size, image_height, image_width, channels] 
  # Note that we've indicated -1 for batch size, 
  # which specifies that this dimension should be
  # dynamically computed based on the number of 
  # input values in features["x"] 
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # The inputs argument specifies our input tensor, 
  # which must have the shape 
  # [batch_size, image_height, image_width, channels]. 
  # Here, we're connecting our first convolutional layer
  # to input_layer, which has the shape 
  # [batch_size, 28, 28, 1].

  # Note: conv2d() will instead accept a shape of 
  # [batch_size, channels, image_height, image_width] 
  # when passed the argument data_format=channels_first.

  # The filters argument specifies the number of filters 
  # to apply (here, 32), and kernel_size specifies the 
  # dimensions of the filters as [height, width] 
  # (here, [5, 5]).

  # TIP: If filter height and width have the same value, 
  # you can instead specify a single integer for 
  # kernel_size—e.g., kernel_size=5.

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # The strides argument specifies the size of the stride. 
  # Here, we set a stride of 2, which indicates that the 
  # subregions extracted by the filter should be separated 
  # by 2 pixels in both the height and width dimensions 
  # (for a 2x2 filter, this means that none of the regions 
  # extracted will overlap). If you want to set different 
  # stride values for height and width, you can instead 
  # specify a tuple or list (e.g., stride=[3, 6]).
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  # we'll flatten our feature map (pool2) to shape 
  # [batch_size, features], so that our tensor has only 
  # two dimensions:
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # The inputs argument specifies the input tensor: 
  # our flattened feature map, pool2_flat. 
  # The units argument specifies the number of neurons in 
  # the dense layer (1,024). The activation argument takes 
  # the activation function; again, we'll use tf.nn.relu 
  # to add ReLU activation. 
  # To help improve the results of our model, 
  # we also apply dropout regularization to our dense layer, 
  # using the dropout method in layers:

  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  
  # The rate argument specifies the dropout rate; 
  # here, we use 0.4, which means 40% of the elements 
  # will be randomly dropped out during training. 
  # The training argument takes a boolean specifying 
  # whether or not the model is currently being run in 
  # training mode; dropout will only be performed if
  # training is True. Here, we check if the mode passed 
  # to our model function cnn_model_fn is TRAIN mode.
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

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
  # Load training and eval data
  # We store the training feature data 
  # (the raw pixel values for 55,000 images of 
  # hand-drawn digits) and training labels 
  # (the corresponding value from 0–9 for each image) 
  # as numpy arrays in train_data and train_labels, 
  # respectively. Similarly, we store the evaluation 
  # feature data (10,000 images) and evaluation labels 
  # in eval_data and eval_labels, respectively.
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  # (a TensorFlow class for performing high-level model 
  # training, evaluation, and inference) for our model

  # The model_fn argument specifies the model function 
  # to use for training, evaluation, and prediction; 
  # we pass it the cnn_model_fn we created 
  # The model_dir argument specifies the directory 
  # where model data (checkpoints) will be saved 
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
      
  # Set up logging for predictions
  # We store a dict of the tensors we want to log in 
  # tensors_to_log. Each key is a label of our choice 
  # that will be printed in the log output, and the 
  # corresponding label is the name of a Tensor in the 
  # TensorFlow graph. Here, our probabilities can be 
  # found in softmax_tensor, the name we gave our 
  # softmax operation earlier when we generated the 
  # probabilities in cnn_model_fn. Note: If you don't 
  # explicitly assign a name to an operation via the 
  # name argument, TensorFlow will assign a default name.
  # A couple easy ways to discover the names applied to 
  # operations are to visualize your graph on TensorBoard) 
  # or to enable the TensorFlow Debugger (tfdbg).

  # after every 50 steps of training.
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  # In the numpy_input_fn call, we pass the training 
  # feature data and labels to x (as a dict) and y, 
  # respectively. We set a batch_size of 100 
  # (which means that the model will train on minibatches 
  # of 100 examples at each step). num_epochs=None means 
  # that the model will train until the specified number 
  # of steps is reached. We also set shuffle=True to 
  # shuffle the training data. In the train call, we 
  # set steps=20000 (which means the model will train 
  # for 20,000 steps total). We pass our logging_hook 
  # to the hooks argument, so that it will be triggered 
  # during training.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

  mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
    
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
