# pipeline.py
import pandas as pd
import unittest
from utils import cargar_datos_prueba, cargar_predicciones, calcular_metricas

# Configuración de rutas de archivos
TEST_FILE = 'Sarcasmo_test.csv'
PREDICTION_FILES = {
    'BERT sin FE': 'predicciones_y_pred.csv',
    'BERT con FE': 'predicciones_y_pred_fe.csv',
    'FastText sin FE': 'predicciones_y_pred_ft_no_fe.csv',
    'FastText con FE': 'predicciones_y_pred_ft_fe.csv'
}

def evaluar_modelos(test_labels, prediction_files):
    resultados = []
    for modelo, file_path in prediction_files.items():
        y_pred = cargar_predicciones(file_path)
        metricas = calcular_metricas(test_labels, y_pred)
        metricas['Modelo'] = modelo
        resultados.append(metricas)
    return pd.DataFrame(resultados)

def guardar_resultados(resultados_df, output_file='resultados_modelos.csv'):
    resultados_df.to_csv(output_file, index=False)
    print(f"Resultados guardados en '{output_file}'")
    print(resultados_df)  # para q imprrima los resultados en la pantalla

# Pruebas unitarias
class TestPipeline(unittest.TestCase):
    def test_cargar_datos_prueba(self):
        test_labels = cargar_datos_prueba(TEST_FILE)
        self.assertTrue(pd.api.types.is_integer_dtype(test_labels), "Las etiquetas de prueba no son de tipo entero.")

    def test_calcular_metricas(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0.1, 0.9, 0.2, 0.8]
        metricas = calcular_metricas(y_true, y_pred)
        self.assertIn('ROC AUC', metricas)
        self.assertIn('Accuracy', metricas)
        self.assertIn('Precision', metricas)


# pruebas unit
if __name__ == '__main__':

    test_labels = cargar_datos_prueba(TEST_FILE)
    
    resultados_df = evaluar_modelos(test_labels, PREDICTION_FILES)
    guardar_resultados(resultados_df)

    unittest.main()
