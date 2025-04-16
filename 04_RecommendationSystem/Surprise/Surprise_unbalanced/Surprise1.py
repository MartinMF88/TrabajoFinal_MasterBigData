import zipfile
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
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
reader = Reader(rating_scale=(0, 1))  # Como 'reordered' es binario (0 o 1)
data = Dataset.load_from_df(df[['user_id', 'product_id', 'reordered']], reader)

# --- PASO 3: Dividir los datos en entrenamiento y prueba ---
trainset, testset = train_test_split(data, test_size=0.2)

# --- PASO 4: Entrenar el modelo SVD ---
model = SVD(n_factors=30, lr_all=0.005, reg_all=0.02)
model.fit(trainset)

# --- PASO 5: Evaluar el modelo en el conjunto de prueba ---
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# Convert predictions to binary values
y_true = [pred.r_ui for pred in predictions]  # Valores reales
y_pred = [1 if pred.est >= 0.5 else 0 for pred in predictions]  # Predicciones binarias

# Calcular Precision, Recall y F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# --- PASO 6: Generar recomendaciones para un usuario específico ---
user_id = 0  # Puedes cambiar el usuario de prueba
all_products = df['product_id'].unique()
predictions = [(item, model.predict(user_id, item).est) for item in all_products]

# Ordenar los productos por puntuación de recomendación
predictions.sort(key=lambda x: x[1], reverse=True)

# --- PASO 7: Mapear los IDs de productos a nombres ---
product_mapping = df[['product_id', 'product_name']].drop_duplicates().set_index('product_id')['product_name']

# Mostrar las 10 mejores recomendaciones
top_recommendations = [product_mapping[item[0]] for item in predictions[:10]]
print("\nRecomendaciones para el usuario 0:")
print(top_recommendations)

#Exportar Datos a csv
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

model_name = 'Surprise1'

save_results(model_name, rmse, precision, recall, f1, results_file)
