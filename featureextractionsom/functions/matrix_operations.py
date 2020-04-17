import numpy.linalg as ln
import numpy as np
import random as rd
from featureextractionsom.type_aliases import Vector, Weight_Matrix
from typing import Tuple, List


def distance(vec1: Vector, vec2: Vector) -> float:
	"""
	Return the Euclidean distance between two vectors
	"""
	return ln.norm(np.subtract(vec1, vec2))


def find_closest(node_matrix: Weight_Matrix, input_vector: Vector, *args) -> Tuple[int, int]:
	"""
	Return row and column coordinates of closest vector in array to given vector
	"""
	# initially set closest distance high
	closest_dist = 0
	closest_row = 0
	closest_col = 0

	for i in range(0, node_matrix.shape[0]):
		# u represents rows -> vectorArray[u] is row u.

		for j in range(0, node_matrix.shape[1]):

			# For evaluation: *args are coordinates of vector within matrix, looking for closest node TO
			# that node so must exclude itself.
			if args:
				if i == args[0] and j == args[1]:
					if args[0] == 0 and args[1] == 0:
						closest_dist = distance(input_vector, node_matrix[1][1])
					continue

			# v represents columns --> vectorArray[u][v] is row u column v

			# Each value of vectorArray[u][v] is a *weight vector*.

			d = distance(input_vector, node_matrix[i][j])

			if i == 0 and j == 0:
				# this is the first iteration, so set "closest" to this value for now.
				closest_dist = d
			else:
				# not the first time through the loop, check if this value of distance is lower than current minimum.
				if d < closest_dist:
					# This distance is smaller! Update distance and row/column values.

					closest_dist = d
					closest_row = i
					closest_col = j

	return closest_row, closest_col


def random_list(length: int) -> List[float]:
	"""
	Return a random list of floats (magnitude between 0 and 1) of given length, for creating the node matrix.
	"""
	return [rd.randint(0, 10) / 10 for i in range(length)]


def build_node_matrix(size: int, n: int) -> Weight_Matrix:
	"""
	returns matrix of nodes for given size
	Each node holds a weight vector of length n, initially populated with random values between 0 and 1
	"""
	# set seed so initial matrix is always the same
	rd.seed(42)

	return np.array([[random_list(n) for column in range(size)] for row in range(size)])


def update_weights(input_weight_vector: Vector, matrix_weight_vector: Vector, scale_factor: float) -> Vector:
	"""
	Update weights in matrix_weight_vector according to formula
	"""

	return matrix_weight_vector + scale_factor * (np.subtract(input_weight_vector, matrix_weight_vector))
