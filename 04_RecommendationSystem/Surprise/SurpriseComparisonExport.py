def generate_tf_comparison_html():
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comparación de Modelos en TensorFlow</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; padding: 20px; background-color: #f9f9f9; }
            h1, h2, h3 { color: #333; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
    </head>
    <body>

        <h1>Comparación de Modelos de TensorFlow</h1>

        <h2>Descripción del Sistema de Recomendación</h2>
        <p>Nuestro sistema de recomendación implementado en TensorFlow utiliza embeddings para modelar las interacciones entre usuarios y productos. Se evaluaron dos versiones del modelo:</p>
        <ul>
            <li><strong>Modelo Original:</strong> Utiliza una arquitectura básica basada en el producto punto de los embeddings de usuario y producto.</li>
            <li><strong>Modelo Tuned:</strong> Incluye mejoras en hiperparámetros y arquitectura, agregando capas densas (MLP) para modelar relaciones más complejas.</li>
        </ul>

        <h2>Comparación de Resultados</h2>
        <table>
            <tr>
                <th>Métrica</th>
                <th>Modelo Original</th>
                <th>Modelo Tuned</th>
            </tr>
            <tr>
                <td>RMSE</td>
                <td>0.4487</td>
                <td><strong>0.3736</strong></td>
            </tr>
            <tr>
                <td>MAE</td>
                <td>0.3579</td>
                <td><strong>0.2899</strong></td>
            </tr>
            <tr>
                <td>Precision@10</td>
                <td>0.7844</td>
                <td><strong>0.8007</strong></td>
            </tr>
            <tr>
                <td>Recall@10</td>
                <td>0.6946</td>
                <td><strong>0.8430</strong></td>
            </tr>
            <tr>
                <td>NDCG@10</td>
                <td>0.2410</td>
                <td><strong>0.8611</strong></td>
            </tr>
        </table>

        <h2>Ejemplo de Recomendaciones</h2>
        <h3>Modelo Original:</h3>
        <ul>
            <li>Packaged poultry (0.01)</li>
            <li>Pickled goods olives (0.01)</li>
            <li>Cream (0.00)</li>
            <li>Yogurt (0.00)</li>
            <li>Fresh fruits (0.00)</li>
        </ul>

        <h3>Modelo Tuned:</h3>
        <ul>
            <li>Spirits (0.74)</li>
            <li>Milk (0.73)</li>
            <li>Bread (0.73)</li>
            <li>Soy lactosefree (0.69)</li>
            <li>Soft drinks (0.69)</li>
        </ul>

        <h2>Conclusión</h2>
        <p>El modelo tuned muestra mejoras significativas en todas las métricas, especialmente en Recall y NDCG@10. Esto sugiere que el modelo no solo hace mejores predicciones, sino que también ordena mejor las recomendaciones. La optimización de hiperparámetros y la inclusión de capas densas han permitido mejorar la calidad del sistema de recomendación de manera notable.</p>

    </body>
    </html>
    """

    output_path = "C:/Users/Florencia/OneDrive/2- MASTER EN BIG DATA/Tesis/Tesis/TrabajoFinal_MasterBigData/04_RecommendationSystem/Tensor_Flow_Recommender/Tensor_Flow_Model_comparison_results.html"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"Archivo HTML guardado en: {output_path}")

# Llamar a la función para generar el HTML
generate_tf_comparison_html()
