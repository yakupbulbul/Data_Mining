from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Step 2: Standardization
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Step 3: Covariance Matrix
covariance_matrix = np.cov(X_std.T)

# Step 4: Eigenvectors and Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Step 5: Principal Components
# Sort the eigenvectors by decreasing eigenvalues
sorted_index = eigenvalues.argsort()[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Select the top k eigenvectors
k = 2
top_k_eigenvectors = sorted_eigenvectors[:, :k]

# Step 6: Projection
X_pca = X_std.dot(top_k_eigenvectors)

# Print the reduced dataset
print(X_pca)
