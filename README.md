# Feature Extraction using Self-Organising Maps
(ST20166622 CIS4013 PRAC1)

## 0.1 Project description
This project aims to prepare a dataset for use in training a Self Organising Map (SOM), and then, using part of the data for training and part for validation and testing, select appropriate parameters to generate and train a matrix of weight vectors to display clusters in response to different observations from the dataset.

The processes and results are stored in this folder (`featureextractionsomproject`) as the following scripts and notebooks:

_This project forms the PRAC1 assessment of module 'CIS4013 Mathematics for Computing' at Cardiff Metropolitan University._

The documentation aspect of PRAC1 is provided in notebook form as outlined below.

## Contents

### Notebooks

#### 01 Data Preparation

In this notebook, the dataset used ('Online Shopper Intention', Sakar et al., 2018) is described and justifications provided for its selection. The data is prepared for analysis and divided into training, validation and testing sets.

#### 02 Theoretical considerations

In this notebook the choices made to develop the SOM are explained and different combinations of functions and parameters considered.

#### 03 Developing the SOM algorithm

The different parameters and functions discussed in Notebook 2 are iteratively adjusted and tuned to obtain the best configuration for  training the weight matrix on the dataset. The functions used to train several versions of the SOM in this notebook are in the folder `featureextractionsom/functions/som_adjustable_parameters`.

#### 04 Evaluating the SOM algorithm

The configuration selected in Notebook 3, saved as a series of funtions in `featureextractionsom/functions/somap`, is used to train a weight vector matrix and the response of the matrix to the testing data is recorded and discussed. The implications for the dataset in context are considered.

### Data

This folder contains the source dataset, 'online_shoppers_intentions.csv', as well as serialised data shared between notebooks:

- colames.pkl - the feature names
- training_data.pkl - the training set
- test_vectors.pkl - the validation set
- testing_data.pkl - the testing set

All of the `.pkl` files are generated by the scripts in Notebook 1.


### `featureextractionsom`

This folder contains the functions used in the notebooks, as well as the file `type_aliases.py`, which contains type aliases used throughout the project.

In the `functions` folder are:

- `data_preparation.py` - functions used to featurise and normalise data
- `evaluation.py` - functions used to generate and store responses to the trained SOM
- `matrix_operations.py` - functions used throughout the project to perform matrix operations including finding the Euclidean Distance between two vectors
- `som_adjustable_parameters.py` - functions used for training the SOM in Notebook 3, with adjustable parameters such as initial value and half life
- `somap.py` - functions used to generate and train a SOM in Notebook 4
- `utils.py` - other functions used in the project


### Notebook Images

This folder contains images referenced in the notbooks

### Output

In this folder response images are saved when testing and evaluting the SOM. All images are stored in folders whose names correspond to the section of the notebook in which they are generated. Selections of the images stored in the Output folder are referenced throughout the notebooks.

## Dependencies

This project was developed in [PyCharm](https://www.jetbrains.com/pycharm/) and [Jupyter Notebook](https://jupyter.org/) using an [Anaconda](https://www.anaconda.com/) virtual environment.

As well as Python 3.7, the project utilises the following packages:

Package | Version
--- | ---
pandas | 1.0.3
pickleshare | 0.7.5
matplotlib | 3.2.1
numpy | 1.18.2
plotly | 4.6.0
jupyter | 1.0.0
jupyter client | 6.0.0
jupyter console | 6.1.0
jupyter core | 4.6.1


## Bibliography

Sakar, Polat, S.O., Katircioglu, M. et al. (2018) 'Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks', _Neural Computing and Applications volume_ (31) 6893–6908, available online at [https://link.springer.com/article/10.1007%2Fs00521-018-3523-0] [Accessed 01/03/2020]

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science.  [Accessed 01/03/2020]

Matous, F. (2015) '3 ways to get more out of your web analytics', _Harvard Business Review_ [online] available at [https://hbr.org/2015/10/3-ways-to-get-more-out-of-your-web-analytics](https://hbr.org/2015/10/3-ways-to-get-more-out-of-your-web-analytics)  [Accessed 05/03/2020]

Builtwith.com (2020) 'Google Analytics Usage Statistics' [online] available at [https://trends.builtwith.com/analytics/Google-Analytics](https://trends.builtwith.com/analytics/Google-Analytics)  [Accessed 13/04/2020]

Fitzpatrick, L. (2019) 'How data analytics impacts small businesses in 2019', _Business.com_ [online] available at [https://www.business.com/articles/the-state-of-data-analytics-in-2019/](https://www.business.com/articles/the-state-of-data-analytics-in-2019/)  [Accessed 05/04/2020]

Krishnavedala (2014) 'Isometric plot of a two dimensional Gaussian function.'. Shared by CC0 license via [wikimedia commons](https://commons.wikimedia.org/wiki/File:Gaussian_2d.svg)  [Accessed 15/04/2020]

Tian, J., Azarian, M. H. & Pecht, M. (2014) 'Anomaly Detection Using Self-Organizing Maps-Based K-Nearest Neighbor Algorithm, in _Proceedings of the European Conference of the Prognostics and Health Management Society_, available online at [SemanticScholar.org](https://www.semanticscholar.org/paper/Anomaly-Detection-Using-Self-Organizing-Maps-Based-Tian-Azarian/0cfcffcf796f0f2f2be202222a07584c9474541c)  [Accessed 04/03/2020]
