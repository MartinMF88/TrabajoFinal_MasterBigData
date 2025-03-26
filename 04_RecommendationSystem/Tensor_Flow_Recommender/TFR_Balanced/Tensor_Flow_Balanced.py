import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import zipfile
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score

# Disable oneDNN optimizations to avoid numerical differences
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Paths
data_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_balanced.csv"
output_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\04_RecommendationSystem\Tensor_Flow_Recommender\TFR_Balanced\Results_Balanced.csv"

# Load data
print("Loading dataset...")
df = pd.read_csv(data_path)
df["product_id"] = df["product_id"].astype(str)

# Ensure unique mapping for product names
if "product_name" in df.columns:
    product_mapping = (
        df[["product_id", "product_name"]]
        .drop_duplicates(subset=["product_id"])
        .set_index("product_id")["product_name"]
    )
else:
    product_mapping = {}

# Convert to TensorFlow dataset
dataset_tf = tf.data.Dataset.from_tensor_slices({
    "user_id": df["user_id"].astype(str),
    "product_id": df["product_id"],
    "reordered": df["reordered"].astype(np.float32)
})

# Create unique vocabularies for users and products
user_ids = df["user_id"].astype(str).unique().tolist()
product_ids = df["product_id"].unique().tolist()

user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
product_lookup = tf.keras.layers.StringLookup(vocabulary=product_ids, mask_token=None)

# Preprocessing function
def preprocess(features):
    return {
        "user_id": user_lookup(features["user_id"]),
        "product_id": product_lookup(features["product_id"])
    }, features["reordered"]

# Prepare dataset
dataset = dataset_tf.map(preprocess).shuffle(10000).batch(256)

# Define recommendation model
class SimpleRecommender(tfrs.Model):
    def __init__(self, user_vocab_size, product_vocab_size, embedding_dim=32):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(user_vocab_size, embedding_dim)
        self.product_embedding = tf.keras.layers.Embedding(product_vocab_size, embedding_dim)
        self.task = tfrs.tasks.Ranking(loss=tf.keras.losses.MeanSquaredError())

    def compute_loss(self, data, training=False):
        features, labels = data
        user_emb = self.user_embedding(features["user_id"])
        prod_emb = self.product_embedding(features["product_id"])
        scores = tf.reduce_sum(user_emb * prod_emb, axis=1)
        return self.task(labels=labels, predictions=scores)

    def call(self, features):
        user_emb = self.user_embedding(features["user_id"])
        prod_emb = self.product_embedding(features["product_id"])
        return tf.reduce_sum(user_emb * prod_emb, axis=1)

# Initialize and train model
model = SimpleRecommender(user_lookup.vocabulary_size(), product_lookup.vocabulary_size())
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
print("Training model...")
model.fit(dataset, epochs=10)

# Generate recommendations for a test user
test_user = "0"
test_products = product_lookup.get_vocabulary()[:10]
test_products = [p for p in test_products if p != "[UNK]"]

user_embedding = model.user_embedding(tf.convert_to_tensor([user_lookup(test_user)]))
product_embeddings = model.product_embedding(
    tf.convert_to_tensor([product_lookup(p) for p in test_products])
)
scores = tf.reduce_sum(user_embedding * product_embeddings, axis=1).numpy()
recommended_products = sorted(zip(test_products, scores), key=lambda x: x[1], reverse=True)

print("Recommendations for user:", test_user)
for product_id_str, score in recommended_products:
    mapped_name = product_mapping.get(product_id_str, product_id_str)
    print(f"Product: {mapped_name}, Score: {score:.2f}")

# Compute evaluation metrics
y_true, y_pred = [], []
for batch in dataset:
    features, labels = batch
    batch_scores = model(features, training=False)
    y_true.extend(labels.numpy())
    y_pred.extend(batch_scores.numpy())

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
precision = precision_score(y_true, [1 if s >= 0.5 else 0 for s in y_pred])
recall = recall_score(y_true, [1 if s >= 0.5 else 0 for s in y_pred])
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Compute NDCG@10
def ndcg_at_k(y_true, y_pred, k=10):
    ideal_dcg = sum([(1 / np.log2(i + 2)) for i in range(min(k, len(y_true)))])
    sorted_indices = np.argsort(y_pred)[::-1]
    dcg = sum([(y_true[i] / np.log2(rank + 2)) for rank, i in enumerate(sorted_indices[:k])])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

ndcg_10 = ndcg_at_k(y_true, y_pred, k=10)

# Save results to CSV with titles
with open(output_path, "a", encoding="utf-8") as f:
    f.write("\nBALANCED MODEL METRICS\n")
    f.write("BASELINE MODEL\n")
    results_df = pd.DataFrame({
        "Metric": ["RMSE","MAE","Precision","Recall","F1 Score","NDCG@10"],
        "Value": [rmse,mae,precision,recall,f1_score,ndcg_10]
    })
    results_df.to_csv(f, index=False, mode='a', header=False)

print("Results saved to:", output_path)
