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
            El modelo optimizado superó al baseline tanto en precisión como en recall, alcanzando valores de {results_df.iloc[1, 2]:.2%} y {results_df.iloc[1, 3]:.2%} respectivamente. Esto indica que el sistema recomienda con mayor relevancia los productos que los usuarios realmente compran.</p>

            <p><strong>2 - Reducción significativa del RMSE:</strong><br>
            El RMSE bajó de {results_df.iloc[0, 1]:.4f} a {results_df.iloc[1, 1]:.4f}, lo que representa una mejora notable en la capacidad del modelo para predecir correctamente.</p>

            <p><strong>3 - Incremento en el F1-score:</strong><br>
            El F1-score del modelo optimizado fue de {results_df.iloc[1, 4]:.4f} frente a {results_df.iloc[0, 4]:.4f} del modelo base, lo que refleja un mejor balance entre precisión y recall.</p>

            <p><strong>4 - Costo computacional mayor pero aceptable:</strong><br>
            El tiempo de entrenamiento del modelo optimizado fue mayor ({results_df.iloc[1, 5]:.2f} segundos frente a {results_df.iloc[0, 5]:.2f}), pero sigue siendo razonable dada la mejora en rendimiento.</p>
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
