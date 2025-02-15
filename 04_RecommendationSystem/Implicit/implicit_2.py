import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "C:\Users\Matias\Desktop\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_items.csv"  # Update with correct path if needed
df = pd.read_csv(file_path)

# Prepare the user-item interaction matrix
user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='reordered', fill_value=0)

# Convert to sparse matrix format
sparse_user_item = csr_matrix(user_item_matrix.values)

# Split data into training and validation sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)
train_sparse = csr_matrix(train_data.values)
test_sparse = csr_matrix(test_data.values)

# Train ALS model
als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
als_model.fit(train_sparse)

# Function to get recommendations
def get_recommendations(user_id, model, user_item_matrix, sparse_data, n=5):
    user_idx = user_item_matrix.index.get_loc(user_id)  # Get user index
    recommendations = model.recommend(user_idx, sparse_data[user_idx], N=n)
    
    # Convert product indices to actual product IDs
    product_ids = [user_item_matrix.columns[i] for i, _ in recommendations]
    scores = [score for _, score in recommendations]
    return list(zip(product_ids, scores))

# Function to evaluate model using Mean Average Precision (MAP@K)
def mean_average_precision_at_k(model, test_sparse, user_item_matrix, k=5):
    map_score = 0
    user_count = test_sparse.shape[0]
    
    for user_idx in range(user_count):
        recommendations = model.recommend(user_idx, test_sparse[user_idx], N=k)
        recommended_products = [user_item_matrix.columns[i] for i, _ in recommendations]
        actual_purchases = set(test_sparse[user_idx].indices)
        hits = sum(1 for prod in recommended_products if prod in actual_purchases)
        map_score += hits / k
    
    return map_score / user_count

# Example: Get top 5 recommendations for a specific user
user_id_example = df['user_id'].iloc[0]  # Replace with actual user ID
recommendations = get_recommendations(user_id_example, als_model, user_item_matrix, sparse_user_item)

# Evaluate model
map_at_5 = mean_average_precision_at_k(als_model, test_sparse, user_item_matrix, k=5)
print("Mean Average Precision at 5:", map_at_5)
print("Recommendations for user:", recommendations)
