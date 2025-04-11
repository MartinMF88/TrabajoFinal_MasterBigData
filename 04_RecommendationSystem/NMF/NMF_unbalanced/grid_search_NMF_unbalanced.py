import pandas as pd
import numpy as np
import time
from itertools import product
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------
# Load dataset
# -------------------------
file_path = "C:\\Users\\Matias\\Desktop\\TrabajoFinal_MasterBigData\\00_Data_Bases\\Cluster5_1_items.csv"
df = pd.read_csv(file_path)

# Prepare user-item interaction matrix
user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='reordered', fill_value=0)

# Convert to sparse matrix
sparse_user_item = csr_matrix(user_item_matrix.values)

# Split data
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)
train_sparse = csr_matrix(train_data.values)
test_sparse = csr_matrix(test_data.values)

# -------------------------
# NMF Training & Evaluation
# -------------------------
def train_evaluate_nmf(n_components, init, solver, beta_loss, max_iter, random_state, model_name, train_sparse, test_sparse):
    print(f"Training {model_name}...")
    start_time = time.time()
    
    nmf_model = NMF(n_components=n_components, init=init, solver=solver, beta_loss=beta_loss,
                    max_iter=max_iter, random_state=random_state)
    
    nmf_model.fit(train_sparse.toarray())
    train_time = time.time() - start_time

    train_pred = np.dot(nmf_model.transform(train_sparse.toarray()), nmf_model.components_)
    test_pred = np.dot(nmf_model.transform(test_sparse.toarray()), nmf_model.components_)
    rmse = np.sqrt(mean_squared_error(test_sparse.toarray(), test_pred))

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

        precision = np.mean(precisions) if precisions else 0.0
        recall = np.mean(recalls) if recalls else 0.0
        return precision, recall

    precision_at_5, recall_at_5 = precision_recall_at_k(nmf_model, test_sparse, k=5)
    f1_score = 2 * (precision_at_5 * recall_at_5) / (precision_at_5 + recall_at_5) if precision_at_5 + recall_at_5 > 0 else 0.0

    return model_name, rmse, precision_at_5, recall_at_5, f1_score, train_time, n_components, init, solver, beta_loss, max_iter

# -------------------------
# Grid Search
# -------------------------
n_components_list = [20, 50, 100]
init_list = ['random', 'nndsvd', 'nndsvda']
solver_list = ['cd', 'mu']
beta_loss_list = ['frobenius', 'kullback-leibler']
max_iter_list = [200, 300]
random_state = 42

param_grid = list(product(n_components_list, init_list, solver_list, beta_loss_list, max_iter_list))

results = []
for idx, (n_components, init, solver, beta_loss, max_iter) in enumerate(param_grid):
    model_name = f"NMF_{idx+1}"
    print(f"🔍 Testing {model_name} with n_components={n_components}, init={init}, solver={solver}, beta_loss={beta_loss}, max_iter={max_iter}")
    try:
        result = train_evaluate_nmf(n_components, init, solver, beta_loss, max_iter,
                                    random_state, model_name, train_sparse, test_sparse)
        results.append(result)
    except Exception as e:
        print(f"❌ Failed on {model_name}: {e}")

# -------------------------
# Save Results
# -------------------------
results_df = pd.DataFrame(results, columns=[
    'Model', 'RMSE', 'Precision@5', 'Recall@5', 'F1-score', 'Training Time',
    'n_components', 'init', 'solver', 'beta_loss', 'max_iter'
])

output_path = "C:\\Users\\Matias\\Desktop\\TrabajoFinal_MasterBigData\\04_RecommendationSystem\\NMF\\NMF_unbalanced\\grid_results_nmf_unbalanced.csv"
results_df.to_csv(output_path, index=False)
print(f"\n✅ Resultados guardados en: {output_path}")

# Save Best Model Hyperparameters
best_result = results_df.sort_values(by="F1-score", ascending=False).iloc[0]
best_result_path = "C:\\Users\\Matias\\Desktop\\TrabajoFinal_MasterBigData\\04_RecommendationSystem\\NMF\\NMF_unbalanced\\best_nmf_ub_hyperparameters.csv"
best_result.to_csv(best_result_path, index=True, header=True)
print(f"🏆 Best model hyperparameters saved to: {best_result_path}")

# Show top 5 models by F1-score
print(results_df.sort_values(by="F1-score", ascending=False).head())
