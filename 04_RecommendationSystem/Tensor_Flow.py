import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import zipfile

# Cargar los datos
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

# Capas de lookup para convertir strings a índices
user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
product_lookup = tf.keras.layers.StringLookup(vocabulary=product_ids, mask_token=None)

# Convertir dataset a formato (input, label)
def preprocess(features):
    return {
        "user_id": user_lookup(features["user_id"]),
        "product_id": product_lookup(features["product_id"])
    }, features["reordered"]

# Aplicar preprocesamiento
dataset = df_tf.map(preprocess).shuffle(10000).batch(256)

class SimpleRecommender(tfrs.Model):
    def __init__(self, user_vocab_size, product_vocab_size, embedding_dim=32):
        super().__init__()

        # Embeddings para usuarios y productos
        self.user_embedding = tf.keras.layers.Embedding(user_vocab_size, embedding_dim)
        self.product_embedding = tf.keras.layers.Embedding(product_vocab_size, embedding_dim)

        # Tarea de ranking con pérdida de ecm
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError()
        )

    def compute_loss(self, data, training=False):
        # Desempaquetamos la tupla: (features, labels)
        features, labels = data
        # Procesamos las características
        user_embeddings = self.user_embedding(features["user_id"])
        product_embeddings = self.product_embedding(features["product_id"])

        # Producto punto entre embeddings para obtener el score de recomendación
        scores = tf.reduce_sum(user_embeddings * product_embeddings, axis=1)

        return self.task(labels=labels, predictions=scores)

# Instanciar el modelo
model = SimpleRecommender(user_lookup.vocabulary_size(), product_lookup.vocabulary_size())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# Entrenar el modelo
model.fit(dataset, epochs=10)

# Simulamos un usuario de prueba
test_user = "0"  # Cambia según sea necesario
test_products = product_lookup.get_vocabulary()[:10]  # Tomamos algunos productos

# Convertimos el usuario a embedding
user_embedding = model.user_embedding(tf.convert_to_tensor([user_lookup(test_user)]))

# Calculamos scores para cada producto
product_embeddings = model.product_embedding(tf.convert_to_tensor([product_lookup(p) for p in test_products]))
scores = tf.reduce_sum(user_embedding * product_embeddings, axis=1).numpy()

# Ordenamos por score y mostramos las recomendaciones
recommended_products = sorted(zip(test_products, scores), key=lambda x: x[1], reverse=True)

print("🔝 Recomendaciones para el usuario:", test_user)
for product, score in recommended_products:
    print(f"Producto: {product}, Score: {score:.2f}")
