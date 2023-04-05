(30 pts) Reduce the dimension of iris dataset using PCA method on WEKA, Python or MATLAB. Show the steps of this method and give brief information for each step in your report. (You can use default parameters in WEKA for this method.)

(20 pts) How many Principal Components (PCs) can be generated maximum? Why?

(20 pts) Which PC is more important, PC1 or PC2? Why?

(30 pts) What is the function of “varianceCovered” parameter on WEKA? Try different values for this parameter such as 1, 0.99, 0.95, 0.90, and 0.70 and apply on this dataset. Explain the results briefly.




- Principal Component Analysis (PCA) on Iris Dataset
In this project, we perform PCA on the Iris dataset, a classic dataset in machine learning and statistics. The Iris dataset contains measurements of the sepal length, sepal width, petal length, and petal width of three species of Iris flowers: Iris setosa, Iris versicolor, and Iris virginica.

The goal of PCA is to reduce the dimensionality of the dataset, while retaining as much of the original variation as possible. This can be useful for data visualization, as well as for machine learning tasks that require input features with low correlation.


- This project requires the following libraries:
 scikit-learn
 pandas
 numpy

- You can install these libraries using pip, by running the following command in the terminal:

pip install scikit-learn pandas numpy




Usage
- To run the PCA code, simply run the following command in the terminal:

python pca_iris.py

This will print the reduced dataset with only two principal components.

- Results
The reduced dataset obtained from PCA can be used for data visualization or for machine learning tasks that require input features with low correlation. In this project, we used PCA to reduce the four-dimensional Iris dataset to a two-dimensional dataset, while retaining most of the original variation.

- References
Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179-188.
Scikit-learn documentation on PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
License
This project is licensed under the MIT License - see the LICENSE file for details.