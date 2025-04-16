import os

def generate_surprise_comparison_html():
    html = """
    <html>
    <head>
        <title>Comparación de Modelos Surprise</title>
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
        <h1>Comparación de Modelos Surprise</h1>
        <p>Este reporte presenta la comparación de distintos modelos de recomendación utilizando la librería <strong>Surprise</strong>.</p>
        
        <h2>Descripción del Sistema de Recomendación</h2>
        <p>La librería <strong>Surprise</strong> (Simple Python Recommendation System Engine) está diseñada específicamente para construir y evaluar sistemas de recomendación. 
        Es útil para generar recomendaciones personalizadas basadas en interacciones previas entre usuarios y productos. En este caso, Surprise es una excelente opción para recomendaciones de compras en supermercados, 
        ya que permite modelar patrones de consumo y predecir productos de interés para cada usuario en función de sus compras anteriores.</p>
        
        <h2>Carga y Preprocesamiento de Datos</h2>
        <p>Se cargan los datos de productos y usuarios, asegurando que estén correctamente estructurados y listos para ser utilizados en los modelos de recomendación.</p>
        
        <h2>Entrenamiento del Modelo</h2>
        <p>Se evaluaron distintas configuraciones de hiperparámetros para entrenar los modelos de Surprise. En particular, se probaron diferentes combinaciones de <strong>n_factors, lr_all</strong> y <strong>reg_all</strong> para optimizar la calidad de las recomendaciones. 
        Además, en el modelo Surprise4 se exploraron otras configuraciones para evaluar su impacto en el rendimiento del sistema.</p>
        
        <h2>Optimización de Hiperparámetros</h2>
        <p>Para encontrar la mejor configuración de los modelos, se ajustaron los siguientes hiperparámetros:</p>
        <ul>
            <li><strong>n_factors:</strong> Dimensión de los factores latentes en el modelo de factorización de matrices.</li>
            <li><strong>lr_all:</strong> Tasa de aprendizaje utilizada para optimizar el modelo.</li>
            <li><strong>reg_all:</strong> Parámetro de regularización para evitar el sobreajuste.</li>
        </ul>
        
        <h2>Evaluación con Métricas</h2>
        <p>Se utilizaron varias métricas para evaluar el desempeño de los modelos:</p>
        <ul>
            <li><strong>RMSE (Root Mean Squared Error):</strong> Mide el error promedio de las predicciones, penalizando más los errores grandes.</li>
            <li><strong>Precision@10:</strong> Evalúa la proporción de productos relevantes dentro de las 10 primeras recomendaciones.</li>
            <li><strong>Recall@10:</strong> Mide la capacidad del modelo para recuperar productos relevantes dentro del top 10.</li>
            <li><strong>F1-score:</strong> Es la media armónica entre la precisión y el recall, proporcionando un balance entre ambos.</li>
        </ul>
        
        <h2>Métricas de Evaluación</h2>
        <table>
            <tr>
                <th>Modelo</th>
                <th>RMSE</th>
                <th>Precision@10</th>
                <th>Recall@10</th>
                <th>F1-score</th>
            </tr>
            <tr>
                <td>Surprise1</td>
                <td>0.4500</td>
                <td>0.6838</td>
                <td>0.7539</td>
                <td>0.7171</td>
            </tr>
            <tr>
                <td>Surprise2</td>
                <td>0.4492</td>
                <td>0.1680</td>
                <td>0.7376</td>
                <td>0.2536</td>
            </tr>
            <tr>
                <td>Surprise3</td>
                <td>0.4499</td>
                <td>0.1678</td>
                <td>0.7394</td>
                <td>0.2535</td>
            </tr>
            <tr>
                <td>Surprise4</td>
                <td>0.5629</td>
                <td>0.6834</td>
                <td>0.7663</td>
                <td>0.6236</td>
            </tr>
        </table>

        <div class="conclusion">
            <h2>Conclusión</h2>
            <p>Los resultados obtenidos muestran que:</p>
            <ul>
                <li><strong>Surprise2</strong> presenta el mejor RMSE, pero su rendimiento en métricas de recomendación como precisión y F1-score es bajo.</li>
                <li><strong>Surprise1</strong> logra el mejor equilibrio entre error de predicción y calidad de recomendaciones, con un alto F1-score (0.7171).</li>
                <li><strong>Surprise4</strong>, a pesar de tener el peor RMSE, muestra buenos resultados en precisión y recall, con un F1-score de 0.6236.</li>
                <li><strong>Surprise3</strong> es similar a Surprise2, con buen RMSE pero pobre rendimiento en precisión y F1-score.</li>
            </ul>
            <p>En resumen, si se prioriza la precisión y calidad de la recomendación, <strong>Surprise1</strong> es el modelo más adecuado. Sin embargo, si se busca minimizar el error de predicción, <strong>Surprise2</strong> o <strong>Surprise3</strong> serían alternativas viables, aunque con menor calidad en las recomendaciones.</p>
        </div>
    </body>
    </html>
    """

    save_path = r"C:\Users\marti\Documents\ORT\TrabajoFinal_MasterBigData\04_RecommendationSystem\Surprise\Surprise_unbalanced\Surprise_Model_Comparison_Results.html"
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ Archivo guardado exitosamente en:\n{save_path}")
    except Exception as e:
        print(f"❌ Error al guardar el archivo: {str(e)}")
    
    return html

generate_surprise_comparison_html()