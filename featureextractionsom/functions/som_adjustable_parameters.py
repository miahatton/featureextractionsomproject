from typing import List, Dict, Tuple
import numpy as np
import random as rd
from featureextractionsom.type_aliases import Matrix, Weight_Matrix
from featureextractionsom.functions.matrix_operations import find_closest, update_weights

"""
These functions are used to train various testing versions of the SOM matrix. Most make use of a dictionary
of hyper parameters containing information about the alpha and theta functions to be used, and the number
of iterations of training to be performed.
"""


def alpha(s: int, initial_val: float, half_life_fraction: float, max_iter: int, alpha_type: str) -> float:
	"""
	Alpha function: determines extent to which node matrix is altered over time
	This function accepts hyper-parameters initial_val, alpha_type, and half_life_fraction which can be adjusted
	to improve performance of the self-organising map.
	"""

	assert alpha_type in ['exp', 'per'], "Alpha type must be one of 'exp' or 'per'."

	if alpha_type == 'exp':
		# This version of the alpha function is an exponential decay with tweak-able half-life
		# Half life is half_life_frac x max iterations
		half_life = half_life_fraction * max_iter
		alpha_val = initial_val * (2 ** (-s / half_life))

	else:
		# This version of the alpha function is based on percentage of iterations that are left
		alpha_val = initial_val * ((max_iter - s) / max_iter)

	return alpha_val


def theta(u: int, v: int, u_prime: int, v_prime: int, gaussian: bool) -> float:
	"""
	Theta function: determines which nodes are altered
	If Gaussian is false - only neighbouring nodes of the closest node are affected
	If Gaussian is true - all nodes are affected but extent of effect varies with Gaussian function centred on
	Best Matching Unit.
	"""
	# Gaussian
	if gaussian:
		x_portion = ((u_prime - u) ** 2) / 2
		y_portion = ((v_prime - v) ** 2) / 2

		theta_value = np.exp(-(x_portion + y_portion))
		return theta_value

	# Not Gaussian

	# if node is a direct neighbour, return 1
	if abs(u - u_prime) <= 1 or abs(v - v_prime) <= 1:
		return 1.0

	# All other nodes return 0
	else:
		return 0


def calculate_scale_factor(s: int, max_iterations: int, theta_val: float,  params: Dict) -> float:
	"""
	Extract the parameters from the 'params' dictionary and use them to calculate the
	value of the alpha and theta functions at the given coordinates u_prime, v_prime
	"""
	# Get alpha parameters
	initial_val = params['initial_val']
	alpha_type = params['alpha_type']
	if alpha_type == 'exp':
		half_life = params['half_life']
	else:
		half_life = 0.0
	alpha_val = alpha(s, initial_val, half_life, max_iterations, alpha_type)
	scale_factor = theta_val * alpha_val
	return scale_factor


def train_matrix(params: Dict, node_matrix: Weight_Matrix, training_data: Matrix) -> Weight_Matrix:
	"""
	Takes a node matrix as input and trains it on training data, returns the trained node matrix
	This function accepts a dictionary of hyper-parameters used to train the matrix.
	"""
	# initialise the counter s as 1
	s = 1

	node_matrix_trained: Weight_Matrix = node_matrix.copy()
	max_iterations = params['max_iterations']

	while s <= max_iterations:

		# Randomly pick index of next input vector
		next_input_index = rd.randint(0, len(training_data) - 1)

		# select input vector and index chosen
		current_input = training_data[next_input_index]

		# find the coordinates u,v of the nearest node
		u, v = find_closest(node_matrix, current_input)

		# Update the node matrix
		for u_prime in range(node_matrix_trained.shape[0]):
			for v_prime in range(node_matrix_trained.shape[1]):

				"""
				Speed up processing by calculating scaling factor first and 
				only updating the node matrix if the scaling factor at this coordinate
				is not equal to 0 (otherwise the update_weights function will return
				the original vector)
				"""

				# Calculate theta function (could be 0)
				theta_val = theta(u, v, u_prime, v_prime, gaussian=params['gaussian'])

				# Continue only if theta != 0
				if theta != 0:	 # use 1e7 because gaussian function could return float64
					scale_factor = calculate_scale_factor(s, max_iterations, theta_val, params)

					# Update weight vector at this coordinate if scaleFactor is not 0
					node_matrix_trained[u_prime,v_prime] = update_weights(current_input,
																		   node_matrix_trained[u_prime,v_prime],
																		   scale_factor)

		# increment s
		s += 1
	return node_matrix_trained
