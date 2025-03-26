import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import zipfile
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score

# Hiperparameters
BEST_EMBEDDING_DIM = 32
BEST_MLP_UNITS = 64
BEST_LR = 0.007175267364206069

# Data loading and preprocessing
zip_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_items.zip"
csv_filename = 'Cluster5_1_items.csv'

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

# Convert relevant columns to string type
df["user_id"] = df["user_id"].astype(str)
df["product_id"] = df["product_id"].astype(str)

# Create a TensorFlow dataset
df_tf = tf.data.Dataset.from_tensor_slices({
    "user_id": df["user_id"],
    "product_id": df["product_id"],
    "reordered": df["reordered"].astype(np.float32)
})

# Vocabularies for users and products
user_ids = df["user_id"].unique().tolist()
product_ids = df["product_id"].unique().tolist()

user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
product_lookup = tf.keras.layers.StringLookup(vocabulary=product_ids, mask_token=None)

def preprocess(features):
    return {
        "user_id": user_lookup(features["user_id"]),
        "product_id": product_lookup(features["product_id"])
    }, features["reordered"]

dataset = df_tf.map(preprocess).shuffle(10000).batch(256)

# Improved model
class FinalRecommender(tfrs.Model):
    def __init__(self, user_vocab_size, product_vocab_size, embedding_dim, mlp_units):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(user_vocab_size, embedding_dim)
        self.product_embedding = tf.keras.layers.Embedding(product_vocab_size, embedding_dim)
        
        self.user_bias = tf.keras.layers.Embedding(user_vocab_size, 1)
        self.product_bias = tf.keras.layers.Embedding(product_vocab_size, 1)
        
        self.interaction_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_units, activation='relu'),
            tf.keras.layers.Dense(mlp_units // 2, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        self.task = tfrs.tasks.Ranking(loss=tf.keras.losses.MeanSquaredError())

    def call(self, features):
        # Embeddings
        user_emb = self.user_embedding(features["user_id"])
        product_emb = self.product_embedding(features["product_id"])
        
        # Biases
        user_bias = tf.squeeze(self.user_bias(features["user_id"]), axis=-1)
        product_bias = tf.squeeze(self.product_bias(features["product_id"]), axis=-1)
        
        # Interacción entre embeddings
        interaction = tf.concat([
            user_emb,
            product_emb,
            tf.abs(user_emb - product_emb),
            user_emb * product_emb
        ], axis=1)
        
        # Final score
        score = tf.squeeze(self.interaction_mlp(interaction), axis=1)
        return score + user_bias + product_bias

    def compute_loss(self, data, training=False):
        features, labels = data
        predictions = self(features)
        return self.task(labels=labels, predictions=predictions)

# Model construction and training
final_model = FinalRecommender(
    user_vocab_size=user_lookup.vocabulary_size(),
    product_vocab_size=product_lookup.vocabulary_size(),
    embedding_dim=BEST_EMBEDDING_DIM,
    mlp_units=BEST_MLP_UNITS
)

# Compile
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BEST_LR))

final_model.build({
    "user_id": tf.TensorShape([None]),
    "product_id": tf.TensorShape([None])
})

dummy_features = {
    "user_id": tf.constant(["dummy_user"]),
    "product_id": tf.constant(["dummy_product"])
}
_ = final_model({
    "user_id": user_lookup(dummy_features["user_id"]),
    "product_id": product_lookup(dummy_features["product_id"])
})

# Training
final_model.fit(dataset, epochs=10)

# Recommendations
test_user = "0" 
test_products = product_lookup.get_vocabulary()

if "[UNK]" in test_products:
    test_products = [p for p in test_products if p != "[UNK]"]

# Create tensors
user_tensor = tf.constant([test_user] * len(test_products))
user_tensor = user_lookup(user_tensor)
product_tensor = product_lookup(tf.constant(test_products))

features_rec = {"user_id": user_tensor, "product_id": product_tensor}
scores = final_model(features_rec, training=False).numpy()

# Sort products by score
recommendations = sorted(zip(test_products, scores), key=lambda x: x[1], reverse=True)

# Map product IDs to names
product_mapping = (
    df[['product_id', 'product_name']]
    .drop_duplicates()
    .set_index('product_id')['product_name']
)

top_10 = recommendations[:10]
top_recommendations = []
for product_id_str, score in top_10:
    product_name = product_mapping.get(product_id_str, product_id_str)
    top_recommendations.append((product_name, score))

print(f"\nRecomendaciones para el usuario: {test_user}")
for product_name, score in top_recommendations:
    print(f"Producto: {product_name}, Score: {score:.2f}")

# Metrics
y_true = []
y_pred = []
for features, labels in dataset:
    preds = final_model(features, training=False)
    y_true.extend(labels.numpy())
    y_pred.extend(preds.numpy())

# Regression
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

# Clasification
threshold = 0.5
y_pred_binary = [1 if pred >= threshold else 0 for pred in y_pred]
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# NDCG@10
def ndcg_at_k(y_true_vals, y_pred_vals, k=10):
    order = np.argsort(y_pred_vals)[::-1]
    y_true_sorted = np.take(y_true_vals, order[:k])
    discounts = np.log2(np.arange(2, 2 + len(y_true_sorted)))
    dcg = np.sum((2**y_true_sorted - 1) / discounts)
    ideal_sorted = np.sort(y_true_vals)[::-1][:k]
    idcg = np.sum((2**ideal_sorted - 1) / discounts)
    return dcg / idcg if idcg > 0 else 0.0

ndcg = ndcg_at_k(np.array(y_true), np.array(y_pred), k=10)

# Save file in CSV
results_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\04_RecommendationSystem\Tensor_Flow_Recommender\TFR_Unbalanced\Results_Unbalanced.csv"

metrics_df = pd.DataFrame({
    "Model Name": ["OPTIMIZED MODEL"],
    "RMSE": [rmse],
    "MAE": [mae],
    "Precision": [precision],
    "Recall": [recall],
    "F1-score": [f1],
    "NDCG@10": [ndcg]
})

# Append
if os.path.exists(results_path):
    metrics_df.to_csv(results_path, mode='a', header=False, index=False)
else:
    metrics_df.to_csv(results_path, index=False)

print(f"\n--- MÉTRICAS FINALES ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"NDCG@10: {ndcg:.4f}")
print(f"\n✓ Métricas guardadas en: {results_path}")
