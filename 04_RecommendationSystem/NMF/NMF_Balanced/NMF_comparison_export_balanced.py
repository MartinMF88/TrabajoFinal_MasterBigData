import os
import pandas as pd

def generate_nmf_comparison_html():
    # Load results from Comparacion_NMF.py output
    results_path = r"C:\\Users\\Matias\\Desktop\\TrabajoFinal_MasterBigData\\04_RecommendationSystem\\NMF\\NMF_Balanced\\nmf_results_balanced.csv"
    
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
            <p><strong>1 - Mejora notable en Precision y Recall:</strong><br>
            El modelo optimizado superó ampliamente al baseline, con una precisión del 51.97% y un recall del 92.80%, lo que indica que no solo recomienda mejor, sino que también recomienda más productos relevantes para los usuarios.</p>

            <p><strong>2 - Reducción significativa del RMSE:</strong><br>
            El RMSE se redujo de 0.0820 a 0.0457, lo que representa una mejora superior al 44% en la capacidad del modelo para predecir correctamente.</p>

            <p><strong>3 - Mayor equilibrio entre precisión y cobertura (F1-score):</strong><br>
            El F1-score del modelo optimizado fue 0.6663 frente a 0.5096 en el baseline, reflejando un mejor balance entre acierto y cobertura en las recomendaciones.</p>

            <p><strong>4 - Reducción del tiempo de entrenamiento:</strong><br>
            A pesar de ser un modelo más preciso y complejo, el tiempo de entrenamiento se redujo a 26.91 segundos frente a los 67.29 del baseline, posiblemente por una mejor elección de hiperparámetros.</p>
        </div>
    </body>
    </html>
    """
    
    save_path = r"C:\\Users\\Matias\\Desktop\\TrabajoFinal_MasterBigData\\04_RecommendationSystem\\NMF\\NMF_Balanced\\NMF_Comparison_Report_balanced.html"
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ Archivo guardado exitosamente en: {save_path}")
    except Exception as e:
        print(f"❌ Error al guardar el archivo: {str(e)}")
    
    return html

# Ejecutar
generate_nmf_comparison_html()
