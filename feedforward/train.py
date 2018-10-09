import sys
import feedforward as ff
import numpy as np
from load_mnist_data import load_mnist_data

def main():
  # Setup network
  architecture = [784, 40, 10]
  weights = ff.initialize_weights(architecture)
  biases = ff.initialize_biases(architecture)

  # Training params
  alpha = 1e-3
  l2_decay = 0.2
  batch_size = 200

  try:
    epochs = int(sys.argv[1])
  except IndexError:
    epochs = 300

  # Load and format training data
  train_X, train_y, test_X, test_y = load_mnist_data()
  train_X = train_X.reshape(60000, 784).T
  train_y = train_y.T[0]
  test_X = test_X.reshape(10000, 784).T
  test_y = test_y.T[0]
  train_y_onehot = np.zeros((10, 60000))
  for i, label in enumerate(train_y):
      train_y_onehot[label][i] = 1

  # Get accuracy prior to training
  pre_accuracy = ff.test_network(test_X, test_y, architecture, weights, biases) * 100
  print("Testing accuracy pre-training on the 10,000 element test set: {:.2f}%".format(pre_accuracy))

  # Start training
  for epoch in range(epochs):

    # Batch the training data
    for batch in range(60000 // batch_size):
      batch_start = batch_size * batch
      batch_end = batch_size * (batch + 1)
      batch_X = np.array([row[batch_start:batch_end] for row in train_X])
      batch_y = np.array([row[batch_start:batch_end] for row in train_y_onehot])

      # Predict, calculate gradients, and update weights
      activations, outputs = ff.feedforward(batch_X, architecture, weights, biases)
      gradients = ff.backpropagation(batch_y, architecture, activations, outputs, weights, biases)
      weights = ff.update_weights(alpha, l2_decay, weights, gradients, activations)
      biases = ff.update_biases(alpha, l2_decay, biases, gradients, activations)

    # Get accuracy post epoch
    train_acc = ff.test_network(train_X, train_y, architecture, weights, biases) * 100
    test_acc = ff.test_network(test_X, test_y, architecture, weights, biases) * 100

    print("Epoch {} / {}".format(epoch+1, epochs))
    print("\t    Training Set Accuracy: {:.2f}%".format(train_acc))
    print("\t     Testing Set Accuracy: {:.2f}%".format(test_acc))

if __name__ == '__main__':
  main()