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
            <li><strong>RMSE (Root Mean Squared Error):</strong> Mide el error promedio de las predicciones, penalizando más los errores grandes. Un valor más bajo indica un mejor desempeño.</li>
            <li><strong>Precision@10:</strong> Evalúa la proporción de productos relevantes dentro de las 10 primeras recomendaciones. Un valor más alto indica que el modelo es más preciso.</li>
            <li><strong>Recall@10:</strong> Mide la capacidad del modelo para recuperar productos relevantes dentro del top 10. Un valor más alto indica que el modelo cubre una mayor proporción de productos relevantes.</li>
            <li><strong>F1-score@10:</strong> Media armónica de Precision@10 y Recall@10, balanceando ambas métricas para obtener un valor representativo del desempeño global del modelo.</li>
        </ul>
        
        <h2>Análisis de Resultados</h2>
        <p>Los modelos fueron evaluados con base en las métricas mencionadas anteriormente. A continuación, se analizan los resultados de cada modelo:</p>
        
        <h3>Surprise1</h3>
        <p>Este modelo presentó un <strong>RMSE</strong> de 0.4229, lo que indica que las predicciones están bastante cerca de los valores reales. Sin embargo, su <strong>Precision@10</strong> de 0.6960 es bastante alta, lo que sugiere que el modelo es relativamente preciso en las primeras recomendaciones, aunque con un <strong>Recall@10</strong> de 0.2142 y un <strong>F1-score@10</strong> de 0.3276, lo que indica que está recuperando una menor proporción de productos relevantes.</p>
        
        <h3>Surprise2</h3>
        <p>El modelo Surprise2 presenta un <strong>RMSE</strong> de 0.4185, similar al de Surprise1, indicando una buena capacidad de predicción. Sin embargo, su <strong>Precision@10</strong> de 0.0806 es notablemente baja, lo que indica que las primeras recomendaciones no son precisas. A pesar de esto, tiene un <strong>Recall@10</strong> de 0.4558, lo que sugiere que recupera una mayor cantidad de productos relevantes, aunque su <strong>F1-score@10</strong> sigue siendo bajo (0.1252), lo que refleja un rendimiento menos equilibrado entre precisión y recall.</p>
        
        <h3>Surprise3</h3>
        <p>Al igual que Surprise2, Surprise3 muestra un <strong>RMSE</strong> similar (0.4190). Sus métricas de <strong>Precision@10</strong> (0.0809) y <strong>Recall@10</strong> (0.4578) son prácticamente idénticas a las de Surprise2, lo que sugiere un rendimiento similar. El <strong>F1-score@10</strong> de 0.1256 es apenas ligeramente mejor que el de Surprise2.</p>
        
        <h3>Surprise4</h3>
        <p>Surprise4 mostró el mejor desempeño en comparación con los demás modelos, con un <strong>RMSE</strong> de 0.4659, lo que indica que las predicciones están relativamente alejadas de los valores reales, pero todavía en un rango razonable. Sin embargo, sus métricas de <strong>Precision@10</strong> (0.7515) y <strong>Recall@10</strong> (0.8394) son excelentes, lo que significa que el modelo está haciendo muy buenas recomendaciones relevantes. El <strong>F1-score@10</strong> de 0.7422 refleja un equilibrio sólido entre precisión y recall, destacando a este modelo como el mejor para recuperaciones precisas y relevantes de productos.</p>
        
        <h3>Surprise4 (2nd Run)</h3>
        <p>El segundo entrenamiento de Surprise4 presentó resultados muy similares a los de la primera ejecución, con un <strong>RMSE</strong> de 0.4660, un <strong>Precision@10</strong> de 0.7514 y un <strong>Recall@10</strong> de 0.8393. El <strong>F1-score@10</strong> también es muy similar (0.7421), lo que confirma la estabilidad y fiabilidad del modelo en ambas ejecuciones.</p>
        
        <h2>Métricas de Evaluación</h2>
        <table>
            <tr>
                <th>Modelo</th>
                <th>RMSE</th>
                <th>Precision@10</th>
                <th>Recall@10</th>
                <th>F1-score@10</th>
            </tr>
            <tr>
                <td>Surprise1</td>
                <td>0.4229</td>
                <td>0.6960</td>
                <td>0.2142</td>
                <td>0.3276</td>
            </tr>
            <tr>
                <td>Surprise2</td>
                <td>0.4185</td>
                <td>0.0806</td>
                <td>0.4558</td>
                <td>0.1252</td>
            </tr>
            <tr>
                <td>Surprise3</td>
                <td>0.4190</td>
                <td>0.0809</td>
                <td>0.4578</td>
                <td>0.1256</td>
            </tr>
            <tr>
                <td>Surprise4</td>
                <td>0.4659</td>
                <td>0.7515</td>
                <td>0.8394</td>
                <td>0.7422</td>
            </tr>
            <tr>
                <td>Surprise4 (2nd Run)</td>
                <td>0.4660</td>
                <td>0.7514</td>
                <td>0.8393</td>
                <td>0.7421</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    save_path = r"C:\Users\marti\Documents\ORT\TrabajoFinal_MasterBigData\04_RecommendationSystem\Surprise\Surprise_balanced\Surprise_Model_Comparison_Results_Balanced.html"
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ Archivo guardado exitosamente en:\n{save_path}")
    except Exception as e:
        print(f"❌ Error al guardar el archivo: {str(e)}")
    
    return html

generate_surprise_comparison_html()