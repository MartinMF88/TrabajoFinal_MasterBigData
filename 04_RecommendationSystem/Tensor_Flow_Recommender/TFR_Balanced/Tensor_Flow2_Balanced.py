import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score

# File paths
csv_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_balanced.csv"
results_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\04_RecommendationSystem\Tensor_Flow_Recommender\TFR_Balanced\Results_Balanced.csv"

# Load dataset
df = pd.read_csv(csv_path)

# Convert columns to string
df["user_id"] = df["user_id"].astype(str)
df["product_id"] = df["product_id"].astype(str)

# Create TensorFlow dataset
df_tf = tf.data.Dataset.from_tensor_slices({
    "user_id": df["user_id"],
    "product_id": df["product_id"],
    "reordered": df["reordered"].astype(np.float32)
})

# Vocabularies
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

# Optimal hyperparameters
BEST_EMBEDDING_DIM = 64
BEST_MLP_UNITS = 32
BEST_LR = 0.009444175100572154

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
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError()
        )
    
    def compute_loss(self, data, training=False):
        features, labels = data
        scores = self(features)
        return self.task(labels=labels, predictions=scores)
    
    def call(self, features):
        user_emb = self.user_embedding(features["user_id"])
        product_emb = self.product_embedding(features["product_id"])
        user_bias = tf.squeeze(self.user_bias(features["user_id"]), axis=-1)
        product_bias = tf.squeeze(self.product_bias(features["product_id"]), axis=-1)
        
        interaction = tf.concat([
            user_emb,
            product_emb,
            tf.abs(user_emb - product_emb),
            user_emb * product_emb
        ], axis=1)
        
        score = tf.squeeze(self.interaction_mlp(interaction), axis=1)
        return score + user_bias + product_bias

final_model = FinalRecommender(
    user_vocab_size=user_lookup.vocabulary_size(),
    product_vocab_size=product_lookup.vocabulary_size(),
    embedding_dim=BEST_EMBEDDING_DIM,
    mlp_units=BEST_MLP_UNITS
)
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BEST_LR))

# Force model construction
dummy_features = {
    "user_id": user_lookup(tf.constant([user_ids[0]])),
    "product_id": product_lookup(tf.constant([product_ids[0]]))
}
_ = final_model(dummy_features)

# Training
final_model.fit(dataset, epochs=10)

# Evaluate model
y_true, y_pred = [], []
for features, labels in dataset:
    preds = final_model(features, training=False)
    y_true.extend(labels.numpy())
    y_pred.extend(preds.numpy())

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
precision = precision_score(y_true, np.array(y_pred) >= 0.5)
recall = recall_score(y_true, np.array(y_pred) >= 0.5)
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def ndcg_at_k(y_true_vals, y_pred_vals, k=10):
    order = np.argsort(y_pred_vals)[::-1]
    y_true_sorted = np.take(y_true_vals, order[:k])
    discounts = np.log2(np.arange(2, 2 + len(y_true_sorted)))
    dcg = np.sum((2**y_true_sorted - 1) / discounts)
    idcg = np.sum((2**np.sort(y_true_vals)[::-1][:k] - 1) / discounts)
    return dcg / idcg if idcg > 0 else 0.0

ndcg = ndcg_at_k(np.array(y_true), np.array(y_pred), k=10)

# Save results
with open(results_path, "a") as f:
    f.write("\nOPTIMIZED MODEL (with tuned hyperparameters)\n")
    f.write(f"RMSE,{rmse:.4f}\n")
    f.write(f"MAE,{mae:.4f}\n")
    f.write(f"Precision,{precision:.4f}\n")
    f.write(f"Recall,{recall:.4f}\n")
    f.write(f"F1-score,{f1:.4f}\n")
    f.write(f"NDCG@10,{ndcg:.4f}\n")

print("Results saved to Results_Balanced.csv")
