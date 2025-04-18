import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import csv

# Cargar dataset
file_path = r"C:\Users\marti\Documents\ORT\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_items.csv"
df = pd.read_csv(file_path)

# Mappings
user_ids = df['user_id'].unique()
product_ids = df['product_id'].unique()
user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
item_mapping = {product_id: idx for idx, product_id in enumerate(product_ids)}
user_inv_mapping = {idx: user_id for user_id, idx in user_mapping.items()}
item_inv_mapping = {idx: product_id for product_id, idx in item_mapping.items()}

# Remap IDs
df['user_id_idx'] = df['user_id'].map(user_mapping)
df['product_id_idx'] = df['product_id'].map(item_mapping)

# Matriz dispersa
sparse_user_item = csr_matrix((df['reordered'], (df['user_id_idx'], df['product_id_idx'])))

# Split train/test (opcional pero recomendable)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_sparse = csr_matrix((train_df['reordered'], (train_df['user_id_idx'], train_df['product_id_idx'])), shape=sparse_user_item.shape)
test_sparse = csr_matrix((test_df['reordered'], (test_df['user_id_idx'], test_df['product_id_idx'])), shape=sparse_user_item.shape)

# Entrenar modelo
als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
als_model.fit(train_sparse)

# Recomendaciones
def get_recommendations(user_id, model, user_mapping, item_inv_mapping, sparse_matrix, n=5):
    user_idx = user_mapping[user_id]
    item_indices, scores = model.recommend(user_idx, sparse_matrix[user_idx], N=n)
    product_ids = [item_inv_mapping[i] for i in item_indices]
    return list(zip(product_ids, scores))

# Métrica RMSE
def calculate_rmse(model, test_df):
    predictions = []
    actuals = []

    for _, row in test_df.iterrows():
        user_idx = row['user_id_idx']
        item_idx = row['product_id_idx']
        score = model.user_factors[user_idx] @ model.item_factors[item_idx]
        predictions.append(score)
        actuals.append(row['reordered'])  # valor binario

    return mean_squared_error(actuals, predictions, squared=False)

# Métricas Precision, Recall, F1
def calculate_precision_recall_f1(model, sparse_matrix, k=5):
    precisions = []
    recalls = []
    f1s = []

    num_users = sparse_matrix.shape[0]

    for user_idx in range(num_users):
        item_indices, _ = model.recommend(user_idx, sparse_matrix[user_idx], N=k)
        recommended = set(item_indices)
        actual = set(sparse_matrix[user_idx].indices)

        true_positives = len(recommended & actual)
        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(actual) if len(actual) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

# Calcular métricas
map_at_5 = calculate_precision_recall_f1(als_model, test_sparse, k=5)[0]  # Precision promedio (ya la tenías como MAP@5)
rmse = calculate_rmse(als_model, test_df)
precision, recall, f1 = calculate_precision_recall_f1(als_model, test_sparse, k=5)

# Mostrar resultados
print("RMSE:", rmse)
print("Precision@5:", precision)
print("Recall@5:", recall)
print("F1-score@5:", f1)

# Exportar métricas a CSV
metrics = {
    "RMSE": [rmse],
    "Precision@5": [precision],
    "Recall@5": [recall],
    "F1-score@5": [f1]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("als_model_metrics.csv", index=False)
print("Métricas exportadas a als_model_metrics.csv")