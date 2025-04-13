import zipfile
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise import accuracy
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv

# --- PASO 1: Cargar los datos ---
zip_path = r"C:\Users\marti\Documents\ORT\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_balanced.zip"
csv_filename = 'Cluster5_1_balanced.csv'

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

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
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

# --- PASO 8: Evaluación con Precision@K, Recall@K y F1-score ---
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

# Capturamos las métricas
precision, recall, f1_score = precision_recall_f1_at_k(predictions, k=10)

# --- PASO 9: Exportar resultados a CSV ---
def save_results(model_name, rmse, precision, recall, f1_score, file_path):
    headers = ['Model Name', 'RMSE', 'Precision', 'Recall', 'F1-score']
    data = [model_name, rmse, precision, recall, f1_score]

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

results_file = r"C:\Users\marti\Documents\ORT\TrabajoFinal_MasterBigData\04_RecommendationSystem\Surprise\Surprise_balanced\surprise_results_balanced.csv"
model_name = 'Surprise4'

# Usar 'f1_score' en lugar de 'f1'
save_results(model_name, rmse, precision, recall, f1_score, results_file)

# --- PASO 10: Exportar métricas a Excel ---
metrics_df = pd.DataFrame({
    'Métrica': ['RMSE', 'MAE', 'Precision@10', 'Recall@10', 'F1-score@10'],
    'Valor': [rmse, mae, precision, recall, f1_score]
})

metrics_df.to_excel('metricas_modelo_SVD.xlsx', index=False)