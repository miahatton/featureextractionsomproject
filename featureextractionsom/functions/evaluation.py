import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple
from featureextractionsom.type_aliases import Vector, Matrix, Weight_Matrix
import random as rd


def record_response(response: Matrix, path, reverse_colourscale=True) -> None:
    """
    Update the dictionary of results and save response as an image
    """
    colourscale = 'Viridis'
    if reverse_colourscale:
        # for a distance matrix, we seek the lowest value (smallest distance)
        colourscale += '_r'
    # For a dot_product matrix, we seek the highest value (greatest projection)

    fig = go.Figure(data=go.Heatmap(z=response, colorscale=colourscale))
    fig.write_image(path)


def generate_dot_matrix(input_vector: Vector, node_matrix: Weight_Matrix, size: int) -> Matrix:
    """
    Generate a matrix of dot products between the input vector and the node matrix
    """
    # Create empty matrix
    response_matrix = np.zeros((size, size))

    # Find the dot product between the input_vector and the vector at each coordinate [i,j]
    for i in range(node_matrix.shape[0]):
        for j in range(node_matrix.shape[1]):
            response_matrix[i][j] = np.dot(input_vector, node_matrix[i][j])

    return response_matrix


def get_test_vectors(validation_set: List[np.ndarray], num: int) -> List[Tuple]:
    """
    Randomly choose 3 vectors from the validation set for testing
    """
    indices = rd.sample(range(validation_set.shape[0]), num)
    random_vectors = [validation_set[i] for i in indices]
    return zip(indices, random_vectors)