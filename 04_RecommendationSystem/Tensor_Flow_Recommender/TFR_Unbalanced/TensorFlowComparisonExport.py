import os

def generate_tf_comparison_html():
    html = """
    <html>
<head>
    <title>Comparación de Modelos TensorFlow</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }
        h1, h2, h3 { color: #333; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #007bff; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .recommendations { font-size: 14px; color: #555; }
        .conclusion { margin-top: 20px; padding: 15px; background: white; }
    </style>
</head>
<body>
    <h1>Comparación de Modelos de TensorFlow</h1>
    <p>Reporte de comparación de dos modelos de recomendación de TensorFlow uno básico y otro optimizado, evaluando métricas clave.</p>
    
    <h2>Descripción del Sistema de Recomendación</h2>
    <p>Este sistema de recomendación utiliza <strong>TensorFlow Recommenders (TFRS)</strong> para implementar un modelo basado en <strong>embeddings</strong> y una red neuronal para capturar interacciones complejas entre usuarios y productos.</p>
    
    <h3>1. Carga y Preprocesamiento de Datos</h3>
    <p>Se cargan los datos de productos y usuarios, asegurando que estén correctamente estructurados.</p>

    <h3>2. Entrenamiento del Modelo con Embeddings</h3>
    <p>Se utilizan embeddings para representar a los usuarios y productos en un espacio de menor dimensión.</p>

    <h3>3. Optimización de Hiperparámetros</h3>
    <p>Se ajustan valores como:</p>
    <ul>
        <li><strong>embedding_dim:</strong> Dimensión de los embeddings.</li>
        <li><strong>mlp_units:</strong> Número de unidades en la red neuronal.</li>
        <li><strong>learning_rate:</strong> Tasa de aprendizaje.</li>
    </ul>

    <h3>4. Evaluación con Métricas</h3>
    <ul>
        <li><strong>RMSE:</strong> Error cuadrático medio.</li>
        <li><strong>MAE:</strong> Error absoluto medio.</li>
        <li><strong>Precision y Recall:</strong> Evaluación de relevancia.</li>
        <li><strong>NDCG@10:</strong> Calidad del ranking.</li>
        <li><strong>F1-score:</strong> Media armónica de precisión y recall.</li>
    </ul>

    <h2>Métricas de Evaluación</h2>
    <table>
        <tr>
            <th>Métrica</th>
            <th>TFR (Baseline)</th>
            <th>TFR2 (Optimized)</th>
        </tr>
        <tr>
            <td>RMSE</td>
            <td>0.4476</td>
            <td>0.3194</td>
        </tr>
        <tr>
            <td>MAE</td>
            <td>0.3566</td>
            <td>0.2353</td>
        </tr>
        <tr>
            <td>Precision</td>
            <td>0.7851</td>
            <td>0.8622</td>
        </tr>
        <tr>
            <td>Recall</td>
            <td>0.6994</td>
            <td>0.8949</td>
        </tr>
        <tr>
            <td>F1-score</td>
            <td>0.7398</td>
            <td>0.8783</td>
        </tr>
        <tr>
            <td>NDCG@10</td>
            <td>0.2037</td>
            <td>1.0</td>
        </tr>
    </table>

    <h2>Conclusiones</h2>
    <div class="conclusion">
        <p><strong>1-Precisión y Recall Mejorados:</strong><br>
        El modelo optimizado logró una <strong>mayor recall (0.8949 vs. 0.6994)</strong> y una mejora en el F1-score (<strong>0.8783 vs. 0.7398</strong>), indicando que recupera productos más relevantes.</p>

        <p><strong>2-Reducción del Error:</strong><br>
        El <strong>RMSE se redujo de 0.4476 a 0.3194</strong> y el <strong>MAE de 0.3566 a 0.2353</strong>, lo que sugiere una mejora en la precisión del modelo.</p>

        <p><strong>3-Mejor Ranking de Recomendaciones (NDCG@10):</strong><br>
        El <strong>NDCG@10 aumentó de 0.2037 a 1.0</strong>, lo que indica una mejor calidad en el orden de las recomendaciones.</p>

        <p><strong>4-Calidad de las Recomendaciones de Productos:</strong><br>
        En el modelo base, los productos recomendados eran menos específicos y con menor confianza. <br>
        Con el modelo optimizado, las recomendaciones incluyen productos más alineados con las preferencias de los usuarios.</p>
    </div>
</body>
</html>
    """
    
    save_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\04_RecommendationSystem\Tensor_Flow_Recommender\TFR_Unbalanced\Tensor_Flow_Model_Comparison_Results.html"
    
    try:
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Guardar el archivo
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        print(f"Archivo guardado exitosamente en:\n{save_path}")
        
    except Exception as e:
        print(f"Error al guardar el archivo: {str(e)}")
    
    return html

generate_tf_comparison_html()
