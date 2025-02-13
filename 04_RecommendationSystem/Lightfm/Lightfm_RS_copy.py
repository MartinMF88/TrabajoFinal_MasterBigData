import numpy as np
import scipy
import pandas as pd
import sklearn
import zipfile
from sklearn.metrics import mean_squared_error
import pandas as pd
import scipy.sparse as sp
from lightfm import LightFM
from scipy.sparse import coo_matrix, csr_matrix

zip_path = '../00_Data_Bases/Cluster5_1_items.zip' 
csv_filename = 'Cluster5_1_items.csv'

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

# Paso 2: Crear índices únicos para usuarios y productos
df['user_index'] = df['user_id'].astype('category').cat.codes
df['product_index'] = df['product_id'].astype('category').cat.codes


# Paso 3: Crear la matriz de interacciones
# Usamos la columna 'reordered' como indicador de interacción
interaction_matrix = coo_matrix(
    (df['reordered'], (df['user_index'], df['product_index'])),
    shape=(df['user_index'].nunique(), df['product_index'].nunique())
)

# Paso 4: Crear características de usuarios (frecuencia por día)
user_features = coo_matrix(
    pd.get_dummies(df[['user_index', 'day']], columns=['day'])
    .groupby('user_index').sum().iloc[:, 1:].values
)


# Paso 5: Crear características de ítems (categoría del producto)
item_features = coo_matrix(
    pd.get_dummies(df[['product_index', 'department']], columns=['department'])
    .groupby('product_index').sum().iloc[:, 1:].values
)


# Paso 6: Convertir las matrices a csr_matrix para permitir indexación
interaction_matrix_csr = interaction_matrix.tocsr()
user_features_csr = user_features.tocsr()
item_features_csr = item_features.tocsr()

# Paso 7: Convertir las matrices a formato float32
interaction_matrix_csr.data = interaction_matrix_csr.data.astype('float32')
user_features_csr.data = user_features_csr.data.astype('float32')
item_features_csr.data = item_features_csr.data.astype('float32')


# Paso 8: Crear y configurar el modelo LightFM
model = LightFM(loss='warp', no_components=30, learning_rate=0.05)

# Paso 9: Reducir los datos a subconjuntos más pequeños (opcional, para prueba inicial)
small_interaction_matrix = interaction_matrix_csr[:1000, :50]
small_user_features = user_features_csr[:1000, :]
small_item_features = item_features_csr[:50, :]


# Paso 10: Entrenar el modelo con el subconjunto reducido
model.fit(
    small_interaction_matrix,
    user_features=small_user_features,
    item_features=small_item_features,
    epochs=1,  # Solo una época para pruebas
    num_threads=1  # Usar un único hilo para evitar problemas de recursos
)




# Paso 11: Hacer predicciones
# Predicciones para un usuario específico (ID = 0 en este caso)
user_id = 0
scores = model.predict(
    user_ids=user_id,
    item_ids=np.arange(small_interaction_matrix.shape[1]),
    user_features=small_user_features,
    item_features=small_item_features
)

# Paso 12: Ordenar recomendaciones y mapear índices a nombres de productos
recommended_items = np.argsort(-scores)
product_mapping = df[['product_index', 'product_name']].drop_duplicates().set_index('product_index')['product_name']
print("Recomendaciones para el usuario:", product_mapping.iloc[recommended_items[:10]])

print("Recomendaciones para el usuario 0:")
print(product_mapping.iloc[recommended_items[:10]])

