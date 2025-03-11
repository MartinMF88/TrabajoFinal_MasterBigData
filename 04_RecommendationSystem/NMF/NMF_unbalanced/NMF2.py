import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
file_path = "C:\\Users\\Matias\\Desktop\\TrabajoFinal_MasterBigData\\00_Data_Bases\\Cluster5_1_items.csv"
df = pd.read_csv(file_path)

# Prepare the user-item interaction matrix
user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='reordered', fill_value=0)

# Convert to sparse matrix format
sparse_user_item = csr_matrix(user_item_matrix.values)

# Split data into training and validation sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)
train_sparse = csr_matrix(train_data.values)
test_sparse = csr_matrix(test_data.values)

# Train NMF model
nmf_model = NMF(n_components=100, init='nndsvda', solver='mu', beta_loss='kullback-leibler', max_iter=500, random_state=42, l1_ratio=0.5)
nmf_model.fit(train_sparse.toarray())

# Function to get recommendations
def get_recommendations(user_id, model, user_item_matrix, sparse_data, n=5):
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User ID {user_id} not found in dataset.")
    
    user_idx = user_item_matrix.index.get_loc(user_id)  # Get user index
    user_factors = nmf_model.transform(sparse_data[user_idx].toarray().reshape(1, -1)).flatten()
    item_factors = nmf_model.components_.T
    scores = np.dot(item_factors, user_factors)
    top_items = np.argsort(scores)[-n:][::-1]
    recommendations = [(user_item_matrix.columns[i], scores[i]) for i in top_items]
    
    return recommendations

# Function to compute RMSE
def compute_rmse(model, train_data, test_data):
    train_pred = np.dot(model.transform(train_data.toarray()), model.components_)
    test_pred = np.dot(model.transform(test_data.toarray()), model.components_)
    rmse = np.sqrt(mean_squared_error(test_data.toarray(), test_pred))
    return rmse

# Function to compute Precision@K and Recall@K
def precision_recall_at_k(model, test_data, k=5):
    precisions, recalls = [], []
    for user_idx in range(test_data.shape[0]):
        actual_purchases = set(test_data[user_idx].indices)
        if not actual_purchases:
            continue
        
        user_factors = model.transform(test_data[user_idx].toarray().reshape(1, -1)).flatten()
        item_factors = model.components_.T
        scores = np.dot(item_factors, user_factors)
        top_k_items = np.argsort(scores)[-k:][::-1]
        recommended_items = set(top_k_items)
        
        hits = len(actual_purchases & recommended_items)
        precisions.append(hits / k)
        recalls.append(hits / len(actual_purchases))
    
    return np.mean(precisions), np.mean(recalls)

# Example: Get top 5 recommendations for a specific user
user_id_example = df['user_id'].iloc[0]

# Ensure user ID exists in matrix
if user_id_example not in user_item_matrix.index:
    raise ValueError(f"User ID {user_id_example} not found in dataset.")

print("User ID Example:", user_id_example)
print("User Item Matrix Shape:", user_item_matrix.shape)
recommendations = get_recommendations(user_id_example, nmf_model, user_item_matrix, sparse_user_item)

# Evaluate RMSE
rmse_score = compute_rmse(nmf_model, train_sparse, test_sparse)
print("RMSE Score:", rmse_score)

# Evaluate Precision@K and Recall@K
precision_at_5, recall_at_5 = precision_recall_at_k(nmf_model, test_sparse, k=5)
print("Precision@5:", precision_at_5)
print("Recall@5:", recall_at_5)

# Display results
print("Recommendations for user:", recommendations)

# Visualizing RMSE Score
plt.bar(['RMSE'], [rmse_score])
plt.ylabel('Score')
plt.title('Root Mean Squared Error (RMSE)')
plt.show()

# Visualizing Precision and Recall
plt.bar(['Precision@5', 'Recall@5'], [precision_at_5, recall_at_5])
plt.ylabel('Score')
plt.title('Precision and Recall at K=5')
plt.show()



