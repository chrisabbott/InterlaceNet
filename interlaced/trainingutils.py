import os
import sys
import json
import numpy as np
import cv2

def test_remove_pixels(x, n):
  # Switch the indices to make modifications more straightforward
  X = x.T

  for start in range(n):
    for image in X:
      a = image[0::n]
      cv2.imshow("Reshaped 1", a.reshape(28,28//n))
      cv2.waitKey(0)
      b = image[1::n]
      cv2.imshow("Reshaped 2", b.reshape(28,28//n))
      cv2.waitKey(0)
      b[:] = 0
      cv2.imshow("Subsampled", image.reshape(28,28))
      cv2.waitKey(0)
      exit()

def subsample_dataset(x, n):
  X = x.T
  processed = []

  for start in range(n):
    subsampled = []

    for image in X:
      subsampled.append(image[start::n])

    processed.append(np.asarray(subsampled).T)

  # Returns in format (:, 60000)
  return processed

def staggered_init(a, b, init="zeros"):
  if init is "zeros":
    supermatrix_a = np.zeros((a.shape[0]*2,a.shape[1]), dtype=a.dtype)
    supermatrix_b = np.zeros((b.shape[0]*2,b.shape[1]), dtype=b.dtype)
  elif init is "ones":
    supermatrix_a = np.ones((a.shape[0]*2,a.shape[1]), dtype=a.dtype)
    supermatrix_b = np.ones((b.shape[0]*2,b.shape[1]), dtype=b.dtype)
  elif init is "random":
    supermatrix_a = np.random.rand(a.shape[0]*2,a.shape[1])
    supermatrix_b = np.random.rand(b.shape[0]*2,b.shape[1])
  else:
    # Use zeros
    supermatrix_a = np.zeros((a.shape[0]*2,a.shape[1]), dtype=a.dtype)
    supermatrix_b = np.zeros((b.shape[0]*2,b.shape[1]), dtype=b.dtype)

  supermatrix_a[0:a.shape[0], 0:a.shape[1]] = a
  supermatrix_b[b.shape[0]:, 0:b.shape[1]] = b
  return (supermatrix_a, supermatrix_b)

def interweave(x, y, transpose=False):
  if transpose:
    a = x.T
    b = y.T
  else:
    a = x
    b = y

  c = np.empty((a.size + b.size,), dtype=a.dtype)
  c[0::2] = a.flatten()
  c[1::2] = b.flatten()
  c = c.reshape((a.shape[0], a.shape[1]*2))

  return c

def average(a, b, axis=0):
  c = np.mean([a,b], axis=axis)
  return c