import os

def generate_tf_comparison_html():
    html = """
    <html>
<head>
    <title>Comparación de Modelos TensorFlow - Datos Balanceados</title>
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
    <h1>Comparación de Modelos de TensorFlow - Datos Balanceados</h1>
    <p>Reporte de comparación de dos modelos de recomendación de TensorFlow en un dataset balanceado, evaluando métricas clave.</p>
    
    <h2>Descripción del Sistema de Recomendación</h2>
    <p>Este sistema de recomendación utiliza <strong>TensorFlow Recommenders (TFRS)</strong> para implementar un modelo basado en <strong>embeddings</strong> y una red neuronal para capturar interacciones complejas entre usuarios y productos.</p>
    
    <h3>1. Carga y Preprocesamiento de Datos</h3>
    <p>Se cargan los datos de productos y usuarios desde un dataset balanceado, asegurando que las distribuciones de interacciones sean equitativas.</p>

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
            <th>TFR Balanced (Baseline)</th>
            <th>TFR2 Balanced (Optimized)</th>
        </tr>
        <tr>
            <td>RMSE</td>
            <td>0.5768</td>
            <td>0.4707</td>
        </tr>
        <tr>
            <td>MAE</td>
            <td>0.4411</td>
            <td>0.3772</td>
        </tr>
        <tr>
            <td>Precision</td>
            <td>0.4089</td>
            <td>0.4368</td>
        </tr>
        <tr>
            <td>Recall</td>
            <td>0.3448</td>
            <td>0.3167</td>
        </tr>
        <tr>
            <td>F1-score</td>
            <td>0.3741</td>
            <td>0.3672</td>
        </tr>
        <tr>
            <td>NDCG@10</td>
            <td>0.1299</td>
            <td>0.8611</td>
        </tr>
    </table>

    <h2>Conclusiones</h2>
    <div class="conclusion">
        <p><strong>1-Reducción del Error:</strong><br>
        El <strong>RMSE se redujo de 0.5768 a 0.4707</strong> y el <strong>MAE de 0.4411 a 0.3772</strong>, lo que indica una mejora en la precisión del modelo.</p>

        <p><strong>2-Impacto en la Precisión y Recall:</strong><br>
        La precisión tuvo una leve mejora (<strong>0.4089 vs. 0.4368</strong>), pero el recall disminuyó (<strong>0.3448 vs. 0.3167</strong>), lo que sugiere un ajuste en la recuperación de elementos relevantes.</p>

        <p><strong>3-Mejor Ranking de Recomendaciones (NDCG@10):</strong><br>
        El modelo optimizado logró una mejora significativa en el ranking de recomendaciones con <strong>NDCG@10 = 0.8611</strong>, lo que indica una mejor organización de los productos recomendados.</p>
    </div>
</body>
</html>
    """
    
    save_path = r"C:\Users\Florencia\OneDrive\2- MASTER EN BIG DATA\Tesis\Tesis\TrabajoFinal_MasterBigData\04_RecommendationSystem\Tensor_Flow_Recommender\TFR_Balanced\Tensor_Flow_Balanced_Model_Comparison_Results.html"
    
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
