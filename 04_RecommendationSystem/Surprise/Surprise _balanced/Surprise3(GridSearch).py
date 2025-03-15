import zipfile
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy

# --- PASO 1: Cargar los datos ---
zip_path = r"C:\Users\marti\Documents\ORT\TrabajoFinal_MasterBigData\00_Data_Bases\Cluster5_1_balanced.zip"
csv_filename = 'Cluster5_1_balanced.csv'

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

# --- PASO 7: Generar recomendaciones para un usuario específico ---
user_id = 0  # Puedes cambiarlo por otro usuario

# Obtener todos los productos únicos
all_products = df['product_id'].unique()

# Predecir la puntuación para cada producto
predictions = [(item, model.predict(user_id, item).est) for item in all_products]

# Ordenar los productos por puntuación de recomendación
predictions.sort(key=lambda x: x[1], reverse=True)

# --- PASO 8: Mapear los IDs de productos a nombres ---
product_mapping = df[['product_id', 'product_name']].drop_duplicates().set_index('product_id')['product_name']

# Mostrar las 10 mejores recomendaciones
top_recommendations = [product_mapping[item[0]] for item in predictions[:10]]
print("\nRecomendaciones para el usuario 0:")
print(top_recommendations)


print("Mejores parámetros encontrados:", best_params)

