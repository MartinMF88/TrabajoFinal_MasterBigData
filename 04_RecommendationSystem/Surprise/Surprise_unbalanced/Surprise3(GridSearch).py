import zipfile
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
import os
import csv

# --- PASO 1: Cargar los datos ---
zip_path = r"C:\Users\marti\Documents\ORT\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_items.zip"
csv_filename = 'Cluster5_1_items.csv'

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

# --- PASO 2: Crear un dataset para Surprise ---
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'reordered']], reader)

# --- PASO 3: Búsqueda de hiperparámetros óptimos ---
param_grid = {
    'n_factors': [20, 30, 50],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
gs.fit(data)

# Obtener los mejores parámetros
best_params = gs.best_params['rmse']
print("Mejores parámetros encontrados:", best_params)

# --- PASO 4: Dividir los datos en entrenamiento y prueba ---
trainset, testset = train_test_split(data, test_size=0.2)

# --- PASO 5: Entrenar el modelo con los mejores hiperparámetros ---
model = SVD(**best_params)
model.fit(trainset)

# --- PASO 6: Evaluar el modelo ---
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# --- PASO 7: Evaluación con Precision@K, Recall@K y F1-score ---
def precision_recall_f1_at_k(predictions, k=10, threshold=0.5):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls, f1_scores = [], [], []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(reverse=True, key=lambda x: x[0])
        top_k = user_ratings[:k]

        num_relevant = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        num_recommended_relevant = sum((true_r >= threshold) for (_, true_r) in top_k)

        precision = num_recommended_relevant / k
        recall = num_recommended_relevant / num_relevant if num_relevant != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print(f'Precision@{k}: {avg_precision:.4f}')
    print(f'Recall@{k}: {avg_recall:.4f}')
    print(f'F1-score@{k}: {avg_f1:.4f}')

    return avg_precision, avg_recall, avg_f1

# Llamar a la función para evaluación
precision, recall, f1 = precision_recall_f1_at_k(predictions, k=10)

# --- PASO 8: Generar recomendaciones para un usuario específico ---
user_id = 0  # Puedes cambiarlo por otro usuario

# Obtener todos los productos únicos
all_products = df['product_id'].unique()

# Predecir la puntuación para cada producto
predictions_user = [(item, model.predict(user_id, item).est) for item in all_products]

# Ordenar los productos por puntuación de recomendación
predictions_user.sort(key=lambda x: x[1], reverse=True)

# --- PASO 9: Mapear los IDs de productos a nombres ---
product_mapping = df[['product_id', 'product_name']].drop_duplicates().set_index('product_id')['product_name']

# Mostrar las 10 mejores recomendaciones
top_recommendations = [product_mapping[item[0]] for item in predictions_user[:10]]
print("\nRecomendaciones para el usuario 0:")
print(top_recommendations)

print("Mejores parámetros encontrados:", best_params)

# --- PASO 10: Exportar resultados a CSV ---
def save_results(model_name, rmse, precision, recall, f1, file_path):
    headers = ['Model Name', 'RMSE', 'Precision', 'Recall', 'F1-score']
    data = [model_name, rmse, precision, recall, f1]

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

results_file = r"C:\Users\marti\Documents\ORT\TrabajoFinal_MasterBigData\04_RecommendationSystem\Surprise\Surprise_unbalanced\surprise_results_unbalanced.csv"
model_name = 'Surprise3'

save_results(model_name, rmse, precision, recall, f1, results_file)