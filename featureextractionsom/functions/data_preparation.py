from typing import List
import pandas as pd


def featurise_categories(data: pd.Series) -> pd.Series:
	"""
	Convert categorical data in a pandas Series to numerical form and return the Series
	"""
	data_values = data.unique()
	categories = {}
	n = 1
	for item in data_values:
		categories[item] = n
		n += 1
	data_featurised = data.replace(categories)
	return data_featurised


def normalise(data: pd.Series) -> List:
	"""
	Iterate through Series of data points and normalise them, returning normalised values as a list
	"""
	maximum = max(data)
	minimum = min(data)
	new_data = []
	for datapoint in data:
		new_data.append((datapoint - minimum) / (maximum - minimum))
	return new_data
