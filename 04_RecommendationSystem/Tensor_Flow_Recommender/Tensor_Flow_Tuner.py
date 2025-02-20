import tensorflow as tf
import tensorflow_recommenders as tfrs
import keras_tuner as kt
import pandas as pd
import numpy as np
import zipfile

# =============================================================================
# CARGA Y PREPROCESAMIENTO DE LOS DATOS
# =============================================================================
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

# =============================================================================
# FUNCIÓN build_model CON CORRECCIÓN (CONSTRUCCIÓN DEL MODELO FORZADA)
# =============================================================================
def build_model(hp):
    embedding_dim = hp.Choice('embedding_dim', [16, 32, 64])
    mlp_units = hp.Choice('mlp_units', [32, 64, 128])
    learning_rate = hp.Float('lr', 1e-3, 1e-2, sampling='log')
    
    class TunedRecommender(tfrs.Model):
        def __init__(self, user_vocab_size, product_vocab_size, embedding_dim, mlp_units):
            super().__init__()
            # Embeddings para usuarios y productos
            self.user_embedding = tf.keras.layers.Embedding(user_vocab_size, embedding_dim)
            self.product_embedding = tf.keras.layers.Embedding(product_vocab_size, embedding_dim)
            # Sesgos para usuarios y productos
            self.user_bias = tf.keras.layers.Embedding(user_vocab_size, 1)
            self.product_bias = tf.keras.layers.Embedding(product_vocab_size, 1)
            # Red neuronal para modelar interacciones complejas
            self.interaction_mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(mlp_units, activation='relu'),
                tf.keras.layers.Dense(mlp_units // 2, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            # Tarea de ranking utilizando error cuadrático medio
            self.task = tfrs.tasks.Ranking(
                loss=tf.keras.losses.MeanSquaredError()
            )
        
        def compute_loss(self, data, training=False):
            features, labels = data
            user_emb = self.user_embedding(features["user_id"])
            product_emb = self.product_embedding(features["product_id"])
            user_bias = tf.squeeze(self.user_bias(features["user_id"]), axis=-1)
            product_bias = tf.squeeze(self.product_bias(features["product_id"]), axis=-1)
            interaction = tf.concat([user_emb, product_emb,
                                     tf.abs(user_emb - product_emb),
                                     user_emb * product_emb], axis=1)
            score = tf.squeeze(self.interaction_mlp(interaction), axis=1)
            score = score + user_bias + product_bias
            return self.task(labels=labels, predictions=score)
        
        def call(self, features):
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

    model = TunedRecommender(
        user_vocab_size=user_lookup.vocabulary_size(),
        product_vocab_size=product_lookup.vocabulary_size(),
        embedding_dim=embedding_dim,
        mlp_units=mlp_units
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    
    # Forzamos la construcción del modelo usando datos dummy.
    # Utilizamos el primer elemento del vocabulario como dato dummy.
    dummy_features = {
        "user_id": tf.constant([user_lookup.get_vocabulary()[0]]),
        "product_id": tf.constant([product_lookup.get_vocabulary()[0]])
    }
    # Convertir las entradas dummy mediante los lookup:
    dummy_features = {
        "user_id": user_lookup(dummy_features["user_id"]),
        "product_id": product_lookup(dummy_features["product_id"])
    }
    _ = model(dummy_features)
    
    return model

# =============================================================================
# CONFIGURACIÓN DEL TUNER CON KERAS TUNER
# =============================================================================
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,           # Ajusta según cuántos experimentos desees probar
    executions_per_trial=1,
    directory='kt_dir',
    project_name='recommender_tuning'
)

# =============================================================================
# DIVISIÓN DEL DATASET EN ENTRENAMIENTO Y VALIDACIÓN
# =============================================================================
# Nota: Ajusta .take() y .skip() según la cantidad de datos disponibles.
train_dataset = dataset.take(10)
val_dataset = dataset.skip(10)

# =============================================================================
# BÚSQUEDA DE HIPERPARÁMETROS
# =============================================================================
tuner.search(train_dataset, validation_data=val_dataset, epochs=5)

# Imprimir los mejores hiperparámetros encontrados
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Mejores hiperparámetros encontrados:")
print("Embedding Dim:", best_hp.get("embedding_dim"))
print("MLP Units:", best_hp.get("mlp_units"))
print("Learning Rate:", best_hp.get("lr"))
