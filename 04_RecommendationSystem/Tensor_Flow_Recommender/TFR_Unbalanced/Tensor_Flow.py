import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import zipfile
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score
import csv


# Data loading and preprocessing
zip_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_items.zip"
csv_filename = 'Cluster5_1_items.csv'

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

df["product_id"] = df["product_id"].astype(str)

# # Create a mapping from IDs to names
if "product_name" in df.columns:
    product_mapping = (
        df[["product_id", "product_name"]]
        .drop_duplicates()
        .set_index("product_id")["product_name"]
    )
else:
    product_mapping = {}

# Convert to TensorFlow dataset
df_tf = tf.data.Dataset.from_tensor_slices({
    "user_id": df["user_id"].astype(str),
    "product_id": df["product_id"],  # Ya es string
    "reordered": df["reordered"].astype(np.float32)
})

# Create unique vocabularies
user_ids = df["user_id"].astype(str).unique().tolist()
product_ids = df["product_id"].unique().tolist()

user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
product_lookup = tf.keras.layers.StringLookup(vocabulary=product_ids, mask_token=None)

# Convert dataset to (input, label) format
def preprocess(features):
    return {
        "user_id": user_lookup(features["user_id"]),
        "product_id": product_lookup(features["product_id"])
    }, features["reordered"]

# Apply preprocessing
dataset = df_tf.map(preprocess).shuffle(10000).batch(256)

class SimpleRecommender(tfrs.Model):
    def __init__(self, user_vocab_size, product_vocab_size, embedding_dim=32):
        super().__init__()

        # Embeddings for users and products
        self.user_embedding = tf.keras.layers.Embedding(user_vocab_size, embedding_dim)
        self.product_embedding = tf.keras.layers.Embedding(product_vocab_size, embedding_dim)

        # Ranking task with MSE loss
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError()
        )

    def compute_loss(self, data, training=False):
        features, labels = data
        user_embeddings = self.user_embedding(features["user_id"])
        product_embeddings = self.product_embedding(features["product_id"])
        scores = tf.reduce_sum(user_embeddings * product_embeddings, axis=1)
        return self.task(labels=labels, predictions=scores)

    # Call method for inference
    def call(self, features):
        user_embeddings = self.user_embedding(features["user_id"])
        product_embeddings = self.product_embedding(features["product_id"])
        return tf.reduce_sum(user_embeddings * product_embeddings, axis=1)

# Instantiate and compile the model
model = SimpleRecommender(user_lookup.vocabulary_size(), product_lookup.vocabulary_size())
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# Train the model
model.fit(dataset, epochs=10)

# Generate recommendations
test_user = "0" 

# Select some products (first 10 from vocabulary)
test_products = product_lookup.get_vocabulary()[:10]

# Remove [UNK] if present
if "[UNK]" in test_products:
    test_products = [p for p in test_products if p != "[UNK]"]

# Convert user to embedding and compute scores for each product
user_embedding = model.user_embedding(tf.convert_to_tensor([user_lookup(test_user)]))
product_embeddings = model.product_embedding(
    tf.convert_to_tensor([product_lookup(p) for p in test_products])
)
scores = tf.reduce_sum(user_embedding * product_embeddings, axis=1).numpy()

# Sort by score and display recommendations
recommended_products = sorted(zip(test_products, scores), key=lambda x: x[1], reverse=True)

print("Recomendaciones para el usuario:", test_user)
for product_id_str, score in recommended_products:
    # Map ID to name
    mapped_name = product_mapping.get(product_id_str, product_id_str)
    print(f"Producto: {mapped_name}, Score: {score:.2f}")

# METRICS: RMSE y MAE
y_true = []
y_pred = []
for batch in dataset:
    features, labels = batch
    batch_scores = model(features, training=False)
    y_true.extend(labels.numpy())
    y_pred.extend(batch_scores.numpy())

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# METRICS: PRECISIÓN y RECALL
threshold = 0.5
y_pred_binary = [1 if score >= threshold else 0 for score in y_pred]
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# METRIC: F1-SCORE
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"F1 Score: {f1_score:.4f}")

# METRIC: NDCG (Normalized Discounted Cumulative Gain)
def ndcg_at_k(y_true_vals, y_pred_vals, k=10):
    order = np.argsort(y_pred_vals)[::-1]
    y_true_sorted = np.take(y_true_vals, order[:k])
    discounts = np.log2(np.arange(2, 2 + len(y_true_sorted)))
    dcg = np.sum((2**y_true_sorted - 1) / discounts)
    ideal_sorted = np.sort(y_true_vals)[::-1][:k]
    idcg = np.sum((2**ideal_sorted - 1) / discounts)
    return dcg / idcg if idcg > 0 else 0.0

user_ground_truth = df[df["reordered"] == 1].groupby("user_id")["product_id"].apply(list).to_dict()
user_ground_truth = {str(k): [str(x) for x in v] for k, v in user_ground_truth.items()}

all_product_ids = [str(p) for p in df["product_id"].unique().tolist()]
test_users = list(user_ground_truth.keys())[:100]

ndcg_scores = []
K = 10
for user in test_users:
    relevant_items = user_ground_truth[user]
    negative_items = list(set(all_product_ids) - set(relevant_items))
    if len(negative_items) > 50:
        sampled_negatives = np.random.choice(negative_items, size=50, replace=False).tolist()
    else:
        sampled_negatives = negative_items
    candidates = list(set(relevant_items + sampled_negatives))
    
    true_relevance = {p: 1 for p in relevant_items}
    true_relevance.update({p: 0 for p in candidates if p not in true_relevance})
    
    user_tensor = user_lookup(tf.constant([str(user)]))
    candidate_tensor = product_lookup(tf.constant([str(c) for c in candidates]))
    
    user_emb = model.user_embedding(user_tensor)
    prod_emb = model.product_embedding(candidate_tensor)
    batch_scores = tf.reduce_sum(user_emb * prod_emb, axis=1).numpy()
    
    relevance_array = np.array([true_relevance[p] for p in candidates])
    ndcg_val = ndcg_at_k(relevance_array, batch_scores, k=K)
    ndcg_scores.append(ndcg_val)

mean_ndcg = np.mean(ndcg_scores)
print(f"NDCG@{K}: {mean_ndcg:.4f}")

# Define the save path
save_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\04_RecommendationSystem\Tensor_Flow_Recommender\TFR_Unbalanced\Results_Unbalanced.csv"

# Create directories if they don't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Prepare the metrics data
metrics_data = [
    ["UNBALANCED MODEL METRICS"]  # Title
    [""]  # Empty row
    ["BASELINE MODEL"]  # Subtitle
    ["RMSE", f"{rmse:.4f}"],
    ["MAE", f"{mae:.4f}"],
    ["Precision", f"{precision:.4f}"],
    ["Recall", f"{recall:.4f}"],
    ["F1 Score", f"{f1_score:.4f}"],
    ["NDCG@10", f"{mean_ndcg:.4f}"]
]

# Convert to DataFrame and save to CSV
try:
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv(save_path, index=False, header=False, encoding='utf-8')
    print(f"Métricas guardadas exitosamente en: {save_path}")
except Exception as e:
    print(f"Error al guardar el CSV: {e}")

