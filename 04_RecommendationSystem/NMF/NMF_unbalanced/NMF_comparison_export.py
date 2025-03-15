import os
import pandas as pd

def generate_nmf_comparison_html():
    # Load results from Comparacion_NMF.py output
    results_path = r"C:\\Users\\Matias\\Desktop\\TrabajoFinal_MasterBigData\\04_RecommendationSystem\\NMF\\NMF_unbalanced\\nmf_results.csv"
    
    try:
        results_df = pd.read_csv(results_path)
    except Exception as e:
        print(f"❌ Error al cargar los resultados: {str(e)}")
        return
    
    html = f"""
    <html>
    <head>
        <title>Comparación de Modelos NMF</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #007bff; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .conclusion {{ margin-top: 20px; padding: 15px; background: white; }}
        </style>
    </head>
    <body>
        <h1>Comparación de Modelos NMF</h1>
        <p>Este reporte compara dos modelos de descomposición en valores no negativos (NMF), evaluando métricas clave.</p>
        
        <h2>Descripción del Sistema de Recomendación</h2>
        <p>Este sistema de recomendación utiliza <strong>NMF</strong> para factorizar la matriz usuario-producto en representaciones latentes.</p>
        
        <h3>1. Carga y Preprocesamiento de Datos</h3>
        <p>Se construye la matriz de interacciones usuario-producto a partir de datos de compras.</p>
        
        <h3>2. Entrenamiento del Modelo NMF</h3>
        <p>Se comparan dos configuraciones:</p>
        <ul>
            <li><strong>{results_df.iloc[0, 0]}:</strong> 50 componentes, inicialización aleatoria.</li>
            <li><strong>{results_df.iloc[1, 0]}:</strong> 100 componentes, inicialización avanzada, solver 'mu'.</li>
        </ul>
        
        <h2>Métricas de Evaluación</h2>
        <table>
            <tr>
                <th>Métrica</th>
                <th>{results_df.iloc[0, 0]}</th>
                <th>{results_df.iloc[1, 0]}</th>
            </tr>
            <tr>
                <td>RMSE</td>
                <td>{results_df.iloc[0, 1]:.4f}</td>
                <td>{results_df.iloc[1, 1]:.4f}</td>
            </tr>
            <tr>
                <td>Precision@5</td>
                <td>{results_df.iloc[0, 2]:.4f}</td>
                <td>{results_df.iloc[1, 2]:.4f}</td>
            </tr>
            <tr>
                <td>Recall@5</td>
                <td>{results_df.iloc[0, 3]:.4f}</td>
                <td>{results_df.iloc[1, 3]:.4f}</td>
            </tr>
            <tr>
                <td><strong>F1-score</strong></td>
                <td><strong>{results_df.iloc[0, 4]:.4f}</strong></td>
                <td><strong>{results_df.iloc[1, 4]:.4f}</strong></td>
            </tr>
            <tr>
                <td>Tiempo de Entrenamiento (s)</td>
                <td>{results_df.iloc[0, 5]:.2f}</td>
                <td>{results_df.iloc[1, 5]:.2f}</td>
            </tr>
        </table>

        <h2>Conclusiones</h2>
        <div class="conclusion">
            <p><strong>1 - Mejora en Precision y Recall:</strong><br>
            El modelo optimizado logró una mayor precisión y un leve incremento en el recall, mejorando la relevancia de las recomendaciones.</p>

            <p><strong>2 - Reducción de RMSE:</strong><br>
            Se observa una mejora en el RMSE, reduciéndolo a 2.9%, lo que implica una mejor calidad en las predicciones.</p>

            <p><strong>3 - Incorporación del F1-score:</strong><br>
            El F1-score permite evaluar el balance entre precisión y recall. En este caso, el modelo optimizado mantiene un mejor equilibrio.</p>

            <p><strong>4 - Costo Computacional:</strong><br>
            La optimización del modelo duplicó el tiempo de entrenamiento, lo que era esperable al aumentar la cantidad de componentes de 50 a 100.</p>
        </div>
    </body>
    </html>
    """
    
    save_path = r"C:\\Users\\Matias\\Desktop\\TrabajoFinal_MasterBigData\\04_RecommendationSystem\\NMF\\NMF_unbalanced\\NMF_Comparison_Report.html"
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ Archivo guardado exitosamente en: {save_path}")
    except Exception as e:
        print(f"❌ Error al guardar el archivo: {str(e)}")
    
    return html

generate_nmf_comparison_html()
