import tensorflow as tf
import tensorflow_recommenders as tfrs
import keras_tuner as kt
import pandas as pd
import numpy as np
import os

# Path to the dataset
csv_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_balanced.csv"

# Load dataset
df = pd.read_csv(csv_path)

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

# Preprocessing function
def preprocess(features):
    return {
        "user_id": user_lookup(features["user_id"]),
        "product_id": product_lookup(features["product_id"])
    }, features["reordered"]

# Apply preprocessing, shuffle, and batch the dataset
dataset = df_tf.map(preprocess).shuffle(10000).batch(256)

# Model definition
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
            # Neural network to model interactions
            self.interaction_mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(mlp_units, activation='relu'),
                tf.keras.layers.Dense(mlp_units // 2, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            # Ranking task with MSE loss
            self.task = tfrs.tasks.Ranking(loss=tf.keras.losses.MeanSquaredError())
        
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
    
    # Dummy input to force model build
    dummy_features = {
        "user_id": tf.constant([user_lookup.get_vocabulary()[0]]),
        "product_id": tf.constant([product_lookup.get_vocabulary()[0]])
    }
    dummy_features = {
        "user_id": user_lookup(dummy_features["user_id"]),
        "product_id": product_lookup(dummy_features["product_id"])
    }
    _ = model(dummy_features)
    
    return model

# Configure the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory=r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\04_RecommendationSystem\Tensor_Flow_Recommender\TFR_Balanced",
    project_name='recommender_tuning'
)

# Split dataset into training and validation sets
train_dataset = dataset.take(10)
val_dataset = dataset.skip(10)

# Run hyperparameter tuning
tuner.search(train_dataset, validation_data=val_dataset, epochs=5)

# Get best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_results = {
    "Embedding Dim": best_hp.get("embedding_dim"),
    "MLP Units": best_hp.get("mlp_units"),
    "Learning Rate": best_hp.get("lr")
}

# Print best hyperparameters
print("Best hyperparameters found:")
for key, value in best_results.items():
    print(f"{key}: {value}")

# Path to save results
results_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\04_RecommendationSystem\Tensor_Flow_Recommender\TFR_Balanced\Results_Balanced.csv"

# Save results to CSV
if os.path.exists(results_path):
    with open(results_path, "a") as f:
        f.write("\nTUNER RESULTS\n")
        for key, value in best_results.items():
            f.write(f"{key},{value}\n")
else:
    with open(results_path, "w") as f:
        f.write("TUNER RESULTS\n")
        for key, value in best_results.items():
            f.write(f"{key},{value}\n")

print(f"Tuner results saved in {results_path}")
