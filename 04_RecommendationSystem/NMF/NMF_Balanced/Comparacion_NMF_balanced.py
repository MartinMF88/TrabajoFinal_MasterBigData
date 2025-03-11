import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
file_path = "C:\\Users\\Matias\\Desktop\\TrabajoFinal_MasterBigData\\00_Data_Bases\\Cluster5_1_balanced.csv"
df = pd.read_csv(file_path)

# Prepare the user-item interaction matrix
user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='reordered', fill_value=0)

# Convert to sparse matrix format
sparse_user_item = csr_matrix(user_item_matrix.values)

# Split data into training and validation sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)
train_sparse = csr_matrix(train_data.values)
test_sparse = csr_matrix(test_data.values)

# Function to train and evaluate NMF model
def train_evaluate_nmf(n_components, init, solver, beta_loss, max_iter, random_state, model_name):
    print(f"Training {model_name}...")
    start_time = time.time()
    nmf_model = NMF(n_components=n_components, init=init, solver=solver, beta_loss=beta_loss, max_iter=max_iter, random_state=random_state)
    nmf_model.fit(train_sparse.toarray())
    train_time = time.time() - start_time
    
    # Compute RMSE
    train_pred = np.dot(nmf_model.transform(train_sparse.toarray()), nmf_model.components_)
    test_pred = np.dot(nmf_model.transform(test_sparse.toarray()), nmf_model.components_)
    rmse = np.sqrt(mean_squared_error(test_sparse.toarray(), test_pred))
    
    # Compute Precision@K and Recall@K
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
    
    precision_at_5, recall_at_5 = precision_recall_at_k(nmf_model, test_sparse, k=5)
    
    return model_name, rmse, precision_at_5, recall_at_5, train_time

# Train and compare models
models = [
    (50, 'random', 'cd', 'frobenius', 200, 42, 'NMF (Baseline)'),
    (100, 'nndsvda', 'mu', 'kullback-leibler', 500, 42, 'NMF2 (Optimized)')
]

results = [train_evaluate_nmf(*params) for params in models]

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'Precision@5', 'Recall@5', 'Training Time'])
print(results_df)

# Plot Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# RMSE
axes[0].bar(results_df['Model'], results_df['RMSE'])
axes[0].set_title('RMSE Comparison')
axes[0].set_ylabel('RMSE')

# Precision@5
axes[1].bar(results_df['Model'], results_df['Precision@5'])
axes[1].set_title('Precision@5 Comparison')
axes[1].set_ylabel('Score')

# Recall@5
axes[2].bar(results_df['Model'], results_df['Recall@5'])
axes[2].set_title('Recall@5 Comparison')
axes[2].set_ylabel('Score')

plt.tight_layout()
plt.show()

# Save results to CSV for the HTML report
results_path = "C:\\Users\\Matias\\Desktop\\TrabajoFinal_MasterBigData\\04_RecommendationSystem\\NMF\\NMF_Balanced\\nmf_results_balanced.csv"
results_df.to_csv(results_path, index=False)
print(f"✅ Resultados guardados en: {results_path}")
