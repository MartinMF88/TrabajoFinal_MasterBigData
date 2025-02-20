import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import zipfile
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score

# 1. CARGA DE DATOS Y PREPROCESAMIENTO
zip_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_items.zip"
csv_filename = 'Cluster5_1_items.csv'

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

# Convertir las columnas relevantes a tipo string
df["user_id"] = df["user_id"].astype(str)
df["product_id"] = df["product_id"].astype(str)

# Creamos un dataset de TensorFlow
df_tf = tf.data.Dataset.from_tensor_slices({
    "user_id": df["user_id"],
    "product_id": df["product_id"],
    "reordered": df["reordered"].astype(np.float32)
})

# Vocabularios para usuarios y productos
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

# 2. DEFINICIÓN DEL MODELO (con hiperparámetros "tuneados")
BEST_EMBEDDING_DIM = 16
BEST_MLP_UNITS = 64
BEST_LR = 0.004562235637486822

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
    
    def build(self, input_shape):
        # Marca el modelo como construido
        super().build(input_shape)
    
    def compute_loss(self, data, training=False):
        features, labels = data
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
        score = score + user_bias + product_bias
        return self.task(labels=labels, predictions=score)
    
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
        score = score + user_bias + product_bias
        return score

final_model = FinalRecommender(
    user_vocab_size=user_lookup.vocabulary_size(),
    product_vocab_size=product_lookup.vocabulary_size(),
    embedding_dim=BEST_EMBEDDING_DIM,
    mlp_units=BEST_MLP_UNITS
)
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BEST_LR))

# Forzar la construcción del modelo con datos dummy
dummy_features = {
    "user_id": tf.constant([user_lookup.get_vocabulary()[0]]),
    "product_id": tf.constant([product_lookup.get_vocabulary()[0]])
}
dummy_features = {
    "user_id": user_lookup(dummy_features["user_id"]),
    "product_id": product_lookup(dummy_features["product_id"])
}
_ = final_model(dummy_features)

# =============================================================================
# 3. ENTRENAMIENTO
# =============================================================================
final_model.fit(dataset, epochs=10)

# =============================================================================
# 4. GENERACIÓN DE RECOMENDACIONES
# =============================================================================
test_user = "0"  # Usuario de prueba
test_products = product_lookup.get_vocabulary()

# Eliminar token [UNK] si aparece
if "[UNK]" in test_products:
    test_products = [p for p in test_products if p != "[UNK]"]

# Crear tensores
user_tensor = tf.constant([test_user] * len(test_products))
user_tensor = user_lookup(user_tensor)
product_tensor = product_lookup(tf.constant(test_products))

features_rec = {"user_id": user_tensor, "product_id": product_tensor}
scores = final_model(features_rec, training=False).numpy()

# Ordenar los productos por puntaje
recommendations = sorted(zip(test_products, scores), key=lambda x: x[1], reverse=True)

# Mapear IDs a nombres (df tiene "product_id" y "product_name")
product_mapping = (
    df[['product_id', 'product_name']]
    .drop_duplicates()
    .set_index('product_id')['product_name']
)

top_10 = recommendations[:10]
top_recommendations = []
for product_id_str, score in top_10:
    # Usamos get() para no romper si no encuentra la clave
    product_name = product_mapping.get(product_id_str, product_id_str)
    top_recommendations.append((product_name, score))

print(f"🔝 Recomendaciones para el usuario: {test_user}")
for product_name, score in top_recommendations:
    print(f"Producto: {product_name}, Score: {score:.2f}")

# =============================================================================
# 5. CÁLCULO DE MÉTRICAS
# =============================================================================
y_true = []
y_pred = []
for features, labels in dataset:
    preds = final_model(features, training=False)
    y_true.extend(labels.numpy())
    y_pred.extend(preds.numpy())

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

threshold = 0.5
y_pred_binary = [1 if pred >= threshold else 0 for pred in y_pred]
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)

def ndcg_at_k(y_true_vals, y_pred_vals, k=10):
    order = np.argsort(y_pred_vals)[::-1]
    y_true_sorted = np.take(y_true_vals, order[:k])
    discounts = np.log2(np.arange(2, 2 + len(y_true_sorted)))
    dcg = np.sum((2**y_true_sorted - 1) / discounts)
    ideal_sorted = np.sort(y_true_vals)[::-1][:k]
    idcg = np.sum((2**ideal_sorted - 1) / discounts)
    return dcg / idcg if idcg > 0 else 0.0

ndcg = ndcg_at_k(np.array(y_true), np.array(y_pred), k=10)

print(f"\n--- MÉTRICAS ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"NDCG@10: {ndcg:.4f}")

