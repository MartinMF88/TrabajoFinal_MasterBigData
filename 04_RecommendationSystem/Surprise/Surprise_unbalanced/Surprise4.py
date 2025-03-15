import zipfile
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise import accuracy
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# --- PASO 1: Cargar los datos ---
zip_path = r"C:\Users\marti\Documents\ORT\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_items.zip"
csv_filename = 'Cluster5_1_items.csv'

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

# --- PASO 2: Preprocesamiento de datos ---
df['interaction'] = df['reordered'] + df['days_since_prior_order'] * 0.03 + df['add_to_cart_order'] * 0.015 + df['order_hour_of_day'] * 0.02

# Visualización de la distribución de 'interaction'
sns.histplot(df['interaction'], bins=50)
plt.show()

# --- PASO 3: Crear un dataset para Surprise ---
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'interaction']], reader)

# --- PASO 4: Búsqueda de hiperparámetros óptimos ---
param_grid = {
    'n_factors': [50, 100, 150],
    'lr_all': [0.003, 0.007, 0.01],
    'reg_all': [0.03, 0.07, 0.1]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)  # Reducido cv a 3
gs.fit(data)

# Obtener los mejores parámetros
best_params = gs.best_params['rmse']
print("Mejores parámetros encontrados:", best_params)

# --- PASO 5: Dividir los datos en entrenamiento y prueba ---
trainset = data.build_full_trainset()
testset = trainset.build_testset()

# --- PASO 6: Entrenar el modelo con los mejores hiperparámetros ---
model = SVD(**best_params)
model.fit(trainset)

# --- PASO 7: Evaluar el modelo ---
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

def precision_recall_at_k(predictions, k=10, threshold=0.5):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(reverse=True, key=lambda x: x[0])
        top_k = user_ratings[:k]
        num_relevant = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        num_recommended_relevant = sum((true_r >= threshold) for (_, true_r) in top_k)
        precisions.append(num_recommended_relevant / k)
        recalls.append(num_recommended_relevant / num_relevant if num_relevant != 0 else 0)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    print(f'Precision@{k}: {avg_precision:.4f}')
    print(f'Recall@{k}: {avg_recall:.4f}')

precision_recall_at_k(predictions, k=10)

# --- PASO 8: Generar recomendaciones para un usuario específico ---
user_id = 0  # Puedes cambiarlo
all_products = df['product_id'].unique()
predictions = [(item, model.predict(user_id, item).est) for item in all_products]
predictions.sort(key=lambda x: x[1], reverse=True)

# --- PASO 9: Mapear los IDs de productos a nombres ---
product_mapping = df[['product_id', 'product_name']].drop_duplicates().set_index('product_id')['product_name']
top_recommendations = [product_mapping.get(item[0], "Producto Desconocido") for item in predictions[:10]]

print("\nRecomendaciones para el usuario 0:")
print(top_recommendations)

# --- PASO 10: Mostrar métricas finales ---
print("\n--- MÉTRICAS DE EVALUACIÓN ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
