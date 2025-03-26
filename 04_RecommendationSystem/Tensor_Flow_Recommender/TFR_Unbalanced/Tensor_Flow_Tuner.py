import tensorflow as tf
import tensorflow_recommenders as tfrs
import keras_tuner as kt
import pandas as pd
import numpy as np
import zipfile
import os
import csv

# Data loading and preprocessing
zip_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_items.zip"
csv_filename = 'Cluster5_1_items.csv'

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

# Convert to TensorFlow dataset
df_tf = tf.data.Dataset.from_tensor_slices({
    "user_id": df["user_id"].astype(str),
    "product_id": df["product_id"].astype(str),
    "reordered": df["reordered"].astype(np.float32)
})

# Create unique vocabularies
user_ids = df["user_id"].astype(str).unique().tolist()
product_ids = df["product_id"].astype(str).unique().tolist()

# Lookup layers to convert strings to indices
user_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
product_lookup = tf.keras.layers.StringLookup(vocabulary=product_ids, mask_token=None)

# FPreprocessing function
def preprocess(features):
    return {
        "user_id": user_lookup(features["user_id"]),
        "product_id": product_lookup(features["product_id"])
    }, features["reordered"]

# Apply preprocessing, shuffle, and batch the dataset
dataset = df_tf.map(preprocess).shuffle(10000).batch(256)

# BUILD MODEL FUNCTION WITH CORRECTION
def build_model(hp):
    embedding_dim = hp.Choice('embedding_dim', [16, 32, 64])
    mlp_units = hp.Choice('mlp_units', [32, 64, 128])
    learning_rate = hp.Float('lr', 1e-3, 1e-2, sampling='log')
    
    class TunedRecommender(tfrs.Model):
        def __init__(self, user_vocab_size, product_vocab_size, embedding_dim, mlp_units):
            super().__init__()
            # Embeddings for users and products
            self.user_embedding = tf.keras.layers.Embedding(user_vocab_size, embedding_dim)
            self.product_embedding = tf.keras.layers.Embedding(product_vocab_size, embedding_dim)
            # Bias terms for users and products
            self.user_bias = tf.keras.layers.Embedding(user_vocab_size, 1)
            self.product_bias = tf.keras.layers.Embedding(product_vocab_size, 1)
            # Neural network to model complex interactions
            self.interaction_mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(mlp_units, activation='relu'),
                tf.keras.layers.Dense(mlp_units // 2, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            # Ranking task using mean squared error loss
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
    
    # Force model construction using dummy data.
    dummy_features = {
        "user_id": tf.constant([user_lookup.get_vocabulary()[0]]),
        "product_id": tf.constant([product_lookup.get_vocabulary()[0]])
    }
    # Convert dummy inputs through lookup layers:
    dummy_features = {
        "user_id": user_lookup(dummy_features["user_id"]),
        "product_id": product_lookup(dummy_features["product_id"])
    }
    _ = model(dummy_features)
    
    return model

# Configure the keras tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory=r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\04_RecommendationSystem\Tensor_Flow_Recommender\TFR_Unbalanced",
    project_name='recommender_tuning'
)


# Split the dataset into training and validation sets
train_dataset = dataset.take(10)
val_dataset = dataset.skip(10)

# Hyperparameter search
tuner.search(train_dataset, validation_data=val_dataset, epochs=5)

# Print the best hyperparameters found
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Mejores hiperparámetros encontrados:")
print("Embedding Dim:", best_hp.get("embedding_dim"))
print("MLP Units:", best_hp.get("mlp_units"))
print("Learning Rate:", best_hp.get("lr"))

# Define the save path
save_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\04_RecommendationSystem\Tensor_Flow_Recommender\TFR_Unbalanced\Results_Unbalanced.csv"

# Prepare the data
tuner_results = [
    ["TUNER RESULTS"],  # Title
    [""],  # Empty row
    ["Best Hyperparameters"],  # Subtitle
    ["Embedding Dim", best_hp.get("embedding_dim")],
    ["MLP Units", best_hp.get("mlp_units")],
    ["Learning Rate", best_hp.get("lr")]
]

# Append results to CSV
with open(save_path, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(tuner_results)

print(f"Resultados del tuner guardados en: {save_path}")
