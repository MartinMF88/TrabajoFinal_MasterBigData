# Tensor flow recommender modelo 2
# Modelo de Red Neuronal MLP para modelar interacciones no lineales. 
# Con incorporación de términos de sesgo para usuarios y productos.

import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import zipfile
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score

# CARGA Y PREPROCESAMIENTO DE LOS DATOS
zip_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_items.zip"
csv_filename = 'Cluster5_1_items.csv'

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

# Convertir a dataset de TensorFlow
df_tf = tf.data.Dataset.from_tensor_slices({
    "user_id": df["user_id"].astype(str),
    "product_id": df["product_id"].astype(str),
    "reordered": df["reordered"].astype(np.float32)
})

# Crear vocabularios únicos
user_ids = df["user_id"].astype(str).unique().tolist()
product_ids = df["product_id"].astype(str).unique().tolist()

# Capas de lookup para convertir strings a índices (sin token de desconocido)
user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
product_lookup = tf.keras.layers.StringLookup(vocabulary=product_ids, mask_token=None)

# Función de preprocesamiento para el dataset
def preprocess(features):
    return {
        "user_id": user_lookup(features["user_id"]),
        "product_id": product_lookup(features["product_id"])
    }, features["reordered"]

# Aplicar preprocesamiento, barajar y agrupar en batches
dataset = df_tf.map(preprocess).shuffle(10000).batch(256)

# MODELO 2: MLP + Sesgos para usuarios y productos
class ComplexRecommender(tfrs.Model):
    def __init__(self, user_vocab_size, product_vocab_size, embedding_dim=32):
        super().__init__()
        # Embeddings para usuarios y productos
        self.user_embedding = tf.keras.layers.Embedding(user_vocab_size, embedding_dim)
        self.product_embedding = tf.keras.layers.Embedding(product_vocab_size, embedding_dim)
        # Sesgos para usuarios y productos
        self.user_bias = tf.keras.layers.Embedding(user_vocab_size, 1)
        self.product_bias = tf.keras.layers.Embedding(product_vocab_size, 1)
        # Red neuronal para modelar interacciones complejas
        self.interaction_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        # Tarea de ranking utilizando error cuadrático medio
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError()
        )
    
    def compute_loss(self, data, training=False):
        features, labels = data
        
        # Obtener embeddings y sesgos
        user_emb = self.user_embedding(features["user_id"])
        product_emb = self.product_embedding(features["product_id"])
        user_bias = tf.squeeze(self.user_bias(features["user_id"]), axis=-1)
        product_bias = tf.squeeze(self.product_bias(features["product_id"]), axis=-1)
        
        # Construir una representación combinada de la interacción
        interaction = tf.concat([user_emb, product_emb,
                                 tf.abs(user_emb - product_emb),
                                 user_emb * product_emb], axis=1)
        
        # Calcular el score a partir del MLP y sumarle los sesgos
        score = tf.squeeze(self.interaction_mlp(interaction), axis=1)
        score = score + user_bias + product_bias
        
        return self.task(labels=labels, predictions=score)
    
    def call(self, features):
        # Método para inferencia (mismo proceso que en compute_loss)
        user_emb = self.user_embedding(features["user_id"])
        product_emb = self.product_embedding(features["product_id"])
        user_bias = tf.squeeze(self.user_bias(features["user_id"]), axis=-1)
        product_bias = tf.squeeze(self.product_bias(features["product_id"]), axis=-1)
        interaction = tf.concat([user_emb, product_emb,
                                 tf.abs(user_emb - product_emb),
                                 user_emb * product_emb], axis=1)
        score = tf.squeeze(self.interaction_mlp(interaction), axis=1)
        score = score + user_bias + product_bias
        return score

# COMPILACIÓN Y ENTRENAMIENTO DEL MODELO
complex_model = ComplexRecommender(user_lookup.vocabulary_size(), product_lookup.vocabulary_size())
complex_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
complex_model.fit(dataset, epochs=10)

# GENERAR RECOMENDACIONES PARA UN USUARIO DE PRUEBA
test_user = "0"
test_products = product_lookup.get_vocabulary()[:10] 

# Crear un tensor con el usuario repetido y luego aplicar lookup
user_tensor = tf.constant([test_user] * len(test_products))
user_tensor = user_lookup(user_tensor)

# Crear tensor para los productos: se crea primero un tensor de strings
product_tensor = product_lookup(tf.constant(test_products))

# Preparar el diccionario de features para la inferencia
features = {
    "user_id": user_tensor,
    "product_id": product_tensor
}

scores = complex_model(features, training=False).numpy()

recommended_products = sorted(zip(test_products, scores), key=lambda x: x[1], reverse=True)
print("🔝 Recomendaciones para el usuario:", test_user)
for product, score in recommended_products:
    print(f"Producto: {product}, Score: {score:.2f}")

# CÁLCULO DE MÉTRICAS: RMSE, MAE, PRECISIÓN, RECALL Y NDCG
y_true = []
y_pred = []
for batch in dataset:
    features_batch, labels = batch
    scores_batch = complex_model(features_batch, training=False)
    y_true.extend(labels.numpy())
    y_pred.extend(scores_batch.numpy())

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# PRECISIÓN Y RECALL
threshold = 0.5
y_pred_binary = [1 if score >= threshold else 0 for score in y_pred]
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# FUNCIÓN PARA CALCULAR NDCG@K
def ndcg_at_k(y_true, y_pred, k=10):
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.take(y_true, order[:k])
    discounts = np.log2(np.arange(2, 2 + len(y_true_sorted)))
    dcg = np.sum((2**y_true_sorted - 1) / discounts)
    ideal_sorted = np.sort(y_true)[::-1][:k]
    idcg = np.sum((2**ideal_sorted - 1) / discounts)
    return dcg / idcg if idcg > 0 else 0.0

# Preparar datos para evaluar NDCG@10: 
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
    
    # Diccionario de relevancia: 1 si es relevante, 0 si no
    true_relevance = {p: 1 for p in relevant_items}
    true_relevance.update({p: 0 for p in candidates if p not in true_relevance})
    
    # Preparar tensores para usuario y candidatos
    user_tensor = user_lookup(tf.constant([user]))
    candidate_tensor = product_lookup(tf.constant(candidates))
    features = {
        "user_id": tf.repeat(user_tensor, repeats=len(candidates)),
        "product_id": candidate_tensor
    }
    scores_candidate = complex_model(features, training=False).numpy()
    relevance_array = np.array([true_relevance[p] for p in candidates])
    ndcg = ndcg_at_k(relevance_array, scores_candidate, k=K)
    ndcg_scores.append(ndcg)

mean_ndcg = np.mean(ndcg_scores)
print(f"NDCG@{K}: {mean_ndcg:.4f}")
