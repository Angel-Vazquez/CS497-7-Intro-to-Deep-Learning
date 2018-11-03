import numpy as np
from scipy.sparse import coo_matrix

def softmaxCost(W, b, numClasses, inputSize, decay, data, labels):
  """Computes and returns the (cost, gradient)

  W, b - the weight matrix and bias vector parameters
  numClasses - the number of classes 
  inputSize - the size D of each input vector
  decay - weight decay parameter
  data - the D x N input matrix, where each column data[:,n] corresponds to
         a single sample
  labels - an N x 1 matrix containing the labels corresponding for the input data
  """

  N = data.shape[1]

  groundTruth = coo_matrix((np.ones(N, dtype = np.uint8),
                            (labels, np.arange(N)))).toarray()
  cost = 0;
  dW = np.zeros((numClasses, inputSize))
  db = np.zeros((numClasses, 1))

  ## ---------- YOUR CODE HERE --------------------------------------
  #  Instructions: Compute the cost and gradient for softmax regression.
  #                You need to compute dW, dW, and cost.
  #                The groundTruth matrix might come in handy.

  # This is the cost function from page 86 of lecture4
  Z = W.dot(data) + b
  Z = Z - np.max(Z, axis=0, keepdims=True)
  exp_Z = np.exp(Z)
  sum_emp = np.sum(exp_Z, axis=0, keepdims=True)
  prob = exp_Z / sum_emp
  log_prob = np.log(prob)
  tmp = np.multiply(groundTruth, log_prob)
  cost = (-1.0 / N) * np.sum(tmp)
  cost += (0.5 * decay) * np.sum(W**2)
  

  #This is the gradient function from page 86 of lecture4
  dW = (-1.0 / N) * (groundTruth - prob).dot(data.T)
  dW += decay * W

  #This is gradient b
  db = (-1.0 / N) * np.sum((groundTruth - prob), axis=1, keepdims=True)


  return cost, dW, db


def softmaxPredict(W, b, data):
  """Computes and returns the softmax predictions in the input data.

  W, b - model parameters trained using softmaxTrain,
         a numClasses x D matrix and a numClasses x 1 column vector.
  data - the D x N input matrix, where each column data[:,n] corresponds to
         a single sample.
  """

  #  Your code should produce the prediction matrix pred,
  #  where pred(i) is argmax_c P(c | x(i)).
 
  ## ---------- YOUR CODE HERE ------------------------------------------
  #  Instructions: Compute pred using W and b, assuming that the labels
  #                start from 0.
  Z = W.dot(data) + b
  pred = np.argmax(Z, axis=0)



  # ---------------------------------------------------------------------

  return pred
