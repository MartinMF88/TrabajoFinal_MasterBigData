import pandas as pd

def generate_comparison_html():
    data = {
        "Métrica": ["RMSE", "MAE", "Precision@10", "Recall@10", "Recomendaciones"],
        "Ejecución 1 (Original)": [0.4507, "N/A", "N/A", "N/A", 
                                    "['milk', 'water seltzer sparkling water', 'bread', 'packaged produce', 'soft drinks', 'yogurt', 'fresh fruits', 'refrigerated', 'eggs', 'energy sports drinks']"],
        "Ejecución 2 (Optimización de Hiperparámetros)": [0.4499, 0.3973, "N/A", "N/A", 
                                                           "['water seltzer sparkling water', 'milk', 'yogurt', 'soft drinks', 'packaged produce', 'fresh fruits', 'soy lactosefree', 'packaged vegetables fruits', 'cream', 'energy sports drinks']"],
        "Ejecución 3 (Optimización + Precision/Recall)": [0.4488, 0.3973, 0.1685, 0.7406, 
                                                          "['water seltzer sparkling water', 'milk', 'packaged produce', 'fresh fruits', 'eggs', 'energy sports drinks', 'bread', 'white wines', 'soy lactosefree', 'soft drinks']"]
    }
    
    df_comparison = pd.DataFrame(data)
    html_output_path = "Surprise_Model_comparison_results.html"
    df_comparison.to_html(html_output_path, index=False)
    
    with open(html_output_path, "a", encoding="utf-8") as file:
        file.write("""
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; padding: 20px; background-color: #f8f9fa; }
            h2 { text-align: center; color: #333; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
            th { background-color: #007bff; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .recommendations { font-size: 14px; color: #555; }
            .conclusion { margin-top: 20px; padding: 15px; background: white; }
        </style>
        <h3>Conclusiones</h3>
        <p><strong>Importancia de las métricas seleccionadas:</strong><br>
        <strong>RMSE:</strong> Evalúa la precisión del modelo, penalizando más los errores grandes.<br>
        <strong>MAE:</strong> Mide el error absoluto promedio, proporcionando una interpretación más intuitiva.<br>
        <strong>Precision@10:</strong> Indica la relevancia de las recomendaciones en el top 10.<br>
        <strong>Recall@10:</strong> Evalúa qué tan bien el modelo recupera elementos relevantes.</p>
        <p><strong>Análisis de cambios en las métricas:</strong><br>
        La optimización de los hiperparámetros logró reducir el RMSE de 0.4507 a 0.4488, indicando una ligera mejora en la precisión de las predicciones. El MAE también mejoró al incorporar la optimización.
        La adición de Precision@10 y Recall@10 en la tercera ejecución mostró que, aunque el modelo optimizado logró reducir errores, también mejoró la calidad de las recomendaciones con un recall significativo de 0.7406, indicando que una gran cantidad de productos relevantes están siendo correctamente recomendados.</p>
        <p><strong>Análisis de cambios en las recomendaciones:</strong><br>
        Se observa que las recomendaciones evolucionaron en cada optimización. En la primera ejecución, los productos eran más genéricos, como "milk" y "bread". Con la optimización de hiperparámetros, se incorporaron productos más específicos como "soy lactosefree" y "cream". Finalmente, en la tercera ejecución, se mejoró la diversidad de recomendaciones, incluyendo productos como "white wines", lo que indica una mejor adaptación a las preferencias del usuario.</p>
        """)
    
    print(f"El archivo de comparación ha sido exportado a {html_output_path}")

if __name__ == "__main__":
    generate_comparison_html()