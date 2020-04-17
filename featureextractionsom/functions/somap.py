from typing import List, Dict, Tuple
import numpy as np
import random as rd
from featureextractionsom.type_aliases import Vector, Matrix, Weight_Matrix
from featureextractionsom.functions.matrix_operations import update_weights, find_closest, build_node_matrix


# Functions for training the matrix
def alpha_fixed(s: int) -> float:
	"""
	Alpha function: determines extent to which node matrix is altered over time
	Exponential function with half life of 160 iterations
	"""
	alpha_val = 0.5 * (2 ** (-s / 240))
	return alpha_val


def theta_fixed(u: int, v: int, u_prime: int, v_prime: int) -> float:
	"""
	Theta function: determines which nodes are altered
	Extent of effect varies with Gaussian function centred on
	Best Matching Unit.
	Location of BMU is u, v and location of vector to be altered is u_prime, v_prime.
	"""
	x_portion = ((u_prime - u) ** 2) / 2
	y_portion = ((v_prime - v) ** 2) / 2

	theta_value = np.exp(-(x_portion + y_portion))

	return theta_value


def get_trained_som(training_data: Matrix) -> Weight_Matrix:
	"""
	Takes training data as input and trains a matrix of randomly generated weight vectors on the training data.
	Node matrix is size 15x15 and contains weight vectors of length 18
	Returns the trained node matrix
	"""
	node_matrix = build_node_matrix(15, 18)
	node_matrix_trained: Weight_Matrix = node_matrix.copy()

    # loop through 800 iterations
	for s in range(1,1201):

		# Randomly pick index of next input vector
		next_input_index = rd.randint(0, len(training_data) - 1)

		# select input vector and index chosen
		current_input = training_data[next_input_index]

		# find the coordinates u,v of the nearest node
		u, v = find_closest(node_matrix, current_input)

		# Update the node matrix
		for u_prime in range(node_matrix_trained.shape[0]):
			for v_prime in range(node_matrix_trained.shape[1]):

				# Calculate scale factor
				scale_factor = alpha_fixed(s) * theta_fixed(u, v, u_prime, v_prime)

				# Continue only if scale factor is not ~0
				if abs(scale_factor) > 1e-7:

					# Update weight vector at this coordinate if scaleFactor is not 0
					node_matrix_trained[u_prime, v_prime] = update_weights(current_input,
																		   node_matrix_trained[u_prime, v_prime],
																		   scale_factor)

	return node_matrix_trained
