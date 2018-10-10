import os
import sys
import json
import numpy as np

def sigmoid(x, inverse=False):
  result = 1 / (1 + np.exp(-x))

  if not inverse:
    return result
  else:
    return result * (1 - result)

def initialize_weights(architecture):
  return [np.random.randn(j,i) for i,j in zip(architecture[:-1], architecture[1:])]

def initialize_biases(architecture):
  return [np.random.randn(i,1) for i in architecture[1:]]

def calculate_outputs(X, layer, weights, biases):
  return np.dot(weights[layer], X) + biases[layer]

def feedforward(X, architecture, weights, biases, return_label=False):
  activations = []
  outputs = []
  activations.append(X)
  outputs.append(X)
  tmp = X

  # Number of hidden layers = Total layers - input layer - output layer
  hidden_layers = len(architecture) - 1

  for layer_index in range(hidden_layers):
    tmp = calculate_outputs(tmp, layer_index, weights, biases)
    outputs.append(tmp)
    tmp = sigmoid(tmp)
    activations.append(tmp)

  if return_label:
    return tmp
  else:
    return activations, outputs

def backpropagation(y, architecture, activations, outputs, weights, biases, ensemble=False):
  # Stack the vectors on top of one another if using an ensemble, then calculate error
  if ensemble:
    labels = np.vstack([y,y])
  else:
    labels = y

  # Calculate the error in the output layer
  error = np.multiply(activations[-1] - labels, sigmoid(outputs[-1], inverse=True))

  # Maintain a list of gradients for updates
  gradients = []
  gradients.append(error)

  # Number of hidden layers = Total layers - input layer - output layer
  hidden_layers = len(architecture) - 2

  # Gradient calculation
  for layer in range(hidden_layers):
    layer_weights = weights[-(layer+1)].T
    layer_gradients = gradients[layer]
    updated_gradients = np.multiply(np.dot(layer_weights, layer_gradients), 
                                    sigmoid(outputs[-(layer+2)], inverse=True))
    gradients.append(updated_gradients)

  # Return the reversed list of gradients
  return gradients[::-1]

def update_weights(alpha, l2_decay, weights, gradients, activations):
  new_weights = []

  for weight, gradient, activation in zip(weights, gradients, activations):
    regularization = (alpha * l2_decay) * weight
    new_weight = weight - alpha * np.dot(gradient, activation.T) - regularization
    new_weights.append(new_weight)

  return new_weights

def update_biases(alpha, l2_decay, biases, gradients, activations):
  new_biases = []

  for bias, gradient in zip(biases, gradients):
    regularization = (alpha * l2_decay) * bias
    new_bias = bias - alpha * (np.sum(gradient, axis=1)).reshape(bias.shape) - regularization
    new_biases.append(new_bias)

  return new_biases

def test_network(X, y, architecture, weights, biases, ensemble=False):
  acc = []
  predictions = feedforward(X, architecture, weights, biases, return_label=True).T
  batch_size = predictions.shape[0]

  for i in range(batch_size):
    if not ensemble:
      output = np.argmax(predictions[i])
    else:
      split = np.array_split(predictions[i], 2)
      avg = np.mean(split, axis=0)
      output = np.argmax(avg)
    if output == y[i]:
      acc.append(1)

  return float(len(acc)) / float(batch_size)
