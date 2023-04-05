# Importing libraries
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1 Loading iris dataset
iris_data = load_iris()

# Convert the data to pandas dataframe with the feature names as column headers
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# Step 2 Standardization
# Standardize the data to have ean 0 and variance 1
scaler = StandardScaler()

iris_df_std = scaler.fit_transform(iris_df)

# Step 3: Covariance Matrix
# Calculate the covariance matrix of the standardized data
covariance_matrix = np.cov(iris_df_std.T)


# Step 4: Eigenvectors and Eigenvalues
# Calculate the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Step 5: Principal Components
# Sort the eigenvectors by decreasing eigenvalues
sorted_index = eigenvalues.argsort()[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]


# Select the top n eigenvectors, which correspond to the n principal components
n = 2
top_k_eigenvectors = sorted_eigenvectors[:, :n]

# Step 6: Projection
# Project the standardized data onto the selected principal components to obtain the reduced dataset
iris_df_pca = iris_df_std.dot(top_k_eigenvectors)

# Print the reduced dataset
print(iris_df_pca)