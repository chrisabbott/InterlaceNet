'''
    -> Output classifications are averaged at training time (output layer remains (40x20))
    -> This mimics ensemble averaging without as much additional overhead
'''

import sys
import feedforward as ff
import numpy as np
from load_mnist_data import load_mnist_data
from trainingutils import subsample_dataset, interweave, average, staggered_init

def main():
  # Setup network
  architecture1 = [392, 20, 10]
  weights1 = ff.initialize_weights(architecture1)
  biases1 = ff.initialize_biases(architecture1)

  architecture2 = [392, 20, 10]
  weights2 = ff.initialize_weights(architecture2)
  biases2 = ff.initialize_biases(architecture2)

  # Training params
  alpha = 1e-3
  l2_decay = 0.2
  batch_size = 200

  try:
    epochs = int(sys.argv[1])
  except IndexError:
    epochs = 100

  # Load and format training data
  train_X, train_y, test_X, test_y = load_mnist_data()
  train_X = train_X.reshape(60000, 784).T
  train_y = train_y.T[0]
  test_X = test_X.reshape(10000, 784).T
  test_y = test_y.T[0]
  train_y_onehot = np.zeros((10, 60000))
  for i, label in enumerate(train_y):
      train_y_onehot[label][i] = 1

  subsample_train = subsample_dataset(train_X, 2)
  subnet1_train_X, subnet2_train_X = subsample_train[0], subsample_train[1]
  subsample_test = subsample_dataset(test_X, 2)
  subnet1_test_X, subnet2_test_X = subsample_test[0], subsample_test[1]

  # Get accuracy prior to training
  pre_accuracy1 = ff.test_network(subnet1_test_X, test_y, architecture1, weights1, biases1) * 100
  print("Testing accuracy pre-training on the 10,000 element test set for subnet 1: {:.2f}%".format(pre_accuracy1))
  pre_accuracy2 = ff.test_network(subnet2_test_X, test_y, architecture2, weights2, biases2) * 100
  print("Testing accuracy pre-training on the 10,000 element test set for subnet 2: {:.2f}%".format(pre_accuracy2))

  # Start training
  for epoch in range(epochs):

    # Batch the training data
    for batch in range(60000 // batch_size):
      batch_start = batch_size * batch
      batch_end = batch_size * (batch + 1)
      subnet1_batch_X = np.array([row[batch_start:batch_end] for row in subnet1_train_X])
      subnet2_batch_X = np.array([row[batch_start:batch_end] for row in subnet2_train_X])
      batch_y = np.array([row[batch_start:batch_end] for row in train_y_onehot])

      # Predict, calculate gradients, and update weights
      activations1, outputs1 = ff.feedforward(subnet1_batch_X, architecture1, weights1, biases1)
      activations2, outputs2 = ff.feedforward(subnet2_batch_X, architecture2, weights2, biases2)
      gradients1 = ff.backpropagation(batch_y, architecture1, activations1, outputs1, weights1, biases1)
      gradients2 = ff.backpropagation(batch_y, architecture2, activations2, outputs2, weights2, biases2)
      weights1 = ff.update_weights(alpha, l2_decay, weights1, gradients1, activations1)
      weights2 = ff.update_weights(alpha, l2_decay, weights2, gradients2, activations2)
      biases1 = ff.update_biases(alpha, l2_decay, biases1, gradients1, activations1)
      biases2 = ff.update_biases(alpha, l2_decay, biases2, gradients2, activations2)

    # Get accuracy post epoch
    train1_acc = ff.test_network(subnet1_train_X, train_y, architecture1, weights1, biases1) * 100
    train2_acc = ff.test_network(subnet2_train_X, train_y, architecture2, weights2, biases2) * 100
    test1_acc = ff.test_network(subnet1_test_X, test_y, architecture1, weights1, biases1) * 100
    test2_acc = ff.test_network(subnet2_test_X, test_y, architecture2, weights2, biases2) * 100

    print("Epoch {} / {}".format(epoch+1, epochs))
    print("\t    Subnet 1 Training Set Accuracy: {:.2f}%".format(train1_acc))
    print("\t     Subnet 1 Testing Set Accuracy: {:.2f}%".format(test1_acc))
    print("\t    Subnet 2 Training Set Accuracy: {:.2f}%".format(train2_acc))
    print("\t     Subnet 2 Testing Set Accuracy: {:.2f}%".format(test2_acc))

  # Interweave weights
  architecture3 = [784, 40, 10]
  weights3 = []
  biases3 = []

  # Some confusing shit right here
  supermatrix_weight0_a, supermatrix_weight0_b = staggered_init(weights1[0], weights2[0], init="zeros")
  supermatrix_weight1_a, supermatrix_weight1_b = staggered_init(weights1[1], weights2[1], init="zeros")
  weights3.append(interweave(supermatrix_weight0_a, supermatrix_weight0_b))
  biases3.append(np.vstack((biases1[0], biases2[0])))
  weights3.append(interweave(supermatrix_weight1_a, supermatrix_weight1_b))
  biases3.append(np.vstack((biases1[1], biases2[1])))
  #biases3.append(np.mean([biases1[1], biases2[1]], axis=0))
  
  # Sanity check
  for i in range(len(weights1)):
    print("Weights1: (%s, %s)" % weights1[i].shape)

  for i in range(len(weights2)):
    print("Weights2: (%s, %s)" % weights2[i].shape)

  for i in range(len(weights3)):
    print("Weights3: (%s, %s)" % weights3[i].shape)

  for i in range(len(biases1)):
    print("Biases1: (%s, %s)" % biases1[i].shape)

  for i in range(len(biases2)):
    print("Biases2: (%s, %s)" % biases2[i].shape)

  for i in range(len(biases3)):
    print("Biases3: (%s, %s)" % biases3[i].shape)



  # Get accuracy prior to training
  pre_accuracy3 = ff.test_network(test_X, test_y, architecture3, weights3, biases3, ensemble=True) * 100
  print("Testing accuracy pre-training on the 10,000 element test set for aggregate net: {:.2f}%".format(pre_accuracy3))
  
  # Start training
  for epoch in range(epochs):

    # Batch the training data
    for batch in range(60000 // batch_size):
      batch_start = batch_size * batch
      batch_end = batch_size * (batch + 1)
      batch_X = np.array([row[batch_start:batch_end] for row in train_X])
      batch_y = np.array([row[batch_start:batch_end] for row in train_y_onehot])

      # Predict, calculate gradients, and update weights
      activations3, outputs3 = ff.feedforward(batch_X, architecture3, weights3, biases3)
      gradients3 = ff.backpropagation(batch_y, architecture3, activations3, outputs3, weights3, biases3, ensemble=True)
      weights3 = ff.update_weights(alpha, l2_decay, weights3, gradients3, activations3)
      biases3 = ff.update_biases(alpha, l2_decay, biases3, gradients3, activations3)
      
    # Get accuracy post epoch
    train3_acc = ff.test_network(train_X, train_y, architecture3, weights3, biases3, ensemble=True) * 100
    test3_acc = ff.test_network(test_X, test_y, architecture3, weights3, biases3, ensemble=True) * 100
    
    print("Epoch {} / {}".format(epoch+1, epochs))
    print("\t    Aggregate Net Training Set Accuracy: {:.2f}%".format(train3_acc))
    print("\t     Aggregate Net Testing Set Accuracy: {:.2f}%".format(test3_acc))

if __name__ == '__main__':
  main()