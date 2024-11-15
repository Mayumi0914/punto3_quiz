
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

#funciones a usar en el pipilne
def cargar_datos_prueba(path):
    df_test = pd.read_csv(path, sep=';') 
    df_test['Sarcasmo'] = df_test['Sarcasmo'].map({'No': 0, 'Si': 1})
    return df_test['Sarcasmo'].values

def cargar_predicciones(path):
    df = pd.read_csv(path)
    return df.iloc[:, 0].values  

def calcular_metricas(y_true, y_pred):
    """Calcula solo las métricas de ROC AUC, exactitud y precisión."""
    y_pred = np.array(y_pred)  
    roc_auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    precision = precision_score(y_true, (y_pred >= 0.5).astype(int), zero_division=0)
    return {
        'ROC AUC': roc_auc,
        'Accuracy': accuracy,
        'Precision': precision}