import numpy as np

def computeNumericalGradient(J, theta):
  """ Compute numgrad = computeNumericalGradient(J, theta)

  theta: a matrix of parameters
  J: a function that outputs a real-number and the gradient.
  Calling y = J(theta)[0] will return the function value at theta. 
  """

  # Initialize numgrad with zeros
  numgrad = np.zeros(theta.shape)

  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Implement numerical gradient checking, and return the result in numgrad.  
  # You should write code so that numgrad[i][j] is (the numerical approximation to) the 
  # partial derivative of J with respect to theta[i][j], evaluated at theta.  
  # I.e., numgrad[i][j] should be the (approximately) partial derivative of J with 
  # respect to theta[i][j].
  #                
  # Hint: You will probably want to compute the elements of numgrad one at a time. 

  # Set Epsilon
  epsilon = 0.0001

  # Outer for loop to check across the x-axis
  for i in range(theta.shape[0]):
    # Inner for loop to check across the y-axis
    for j in range(theta.shape[1]):
      # Copy current theta value to min
      theta_min = theta.copy()
      # Subtract min point by epsilon and store
      theta_min[i,j] = theta_min[i,j] - epsilon
      # Not sure
      cost_min, dW, db = J(theta_min)
      # Copy current theta for max
      theta_max = theta.copy()
      # Add max point by epsilon and store
      theta_max[i,j] = theta_max[i,j] + epsilon
      # ?
      cost_max, dW, db = J(theta_max)

      # Final Result for gradient k
      numgrad[i][j] = (cost_max - cost_min) / (2 * epsilon)
      
  ## ---------------------------------------------------------------

  return numgrad
