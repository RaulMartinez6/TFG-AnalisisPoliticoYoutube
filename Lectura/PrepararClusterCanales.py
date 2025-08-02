import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datasets import DatasetDict
import random

tam_cluster = 20

def crear_clusters(grupo):
    clusters = []
    total_videos = len(grupo)
    
    if grupo.iloc[0]['Conjunto'] == 'test':  # Solo para el conjunto 'test'
        cluster_number = 1  # Inicializamos el contador de clusters
        for i in range(0, total_videos, tam_cluster):  # Dividir en bloques de 100
            # Asignar un número de cluster (solo el número, no el nombre completo)
            clusters.append(grupo.iloc[i:i+tam_cluster].assign(Cluster=cluster_number))
            cluster_number += 1  # Incrementar el número del cluster
        
        # Concatenar los clusters y devolver el DataFrame actualizado
        return pd.concat(clusters)
    
    return grupo  # Para el conjunto 'train' no creamos clusters

dataset = pd.read_csv('videos.csv')

dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

canal_ideologias = dataset.groupby("idcanal")["Ideologia"].first().reset_index()

train_canales = []
test_canales = []
for ideologia in canal_ideologias["Ideologia"].unique():
    canales_ideologia = canal_ideologias[canal_ideologias["Ideologia"] == ideologia]["idcanal"].tolist()
    
    test = [random.choice(canales_ideologia)]

    if test[0] in ['@AlanBarrosoA', '@sumar_oficial']:
        for canal_especial in ['@AlanBarrosoA', '@sumar_oficial']:
            if canal_especial in canales_ideologia and canal_especial not in test:
                test.append(canal_especial)

    train = [c for c in canales_ideologia if c not in test]
    train_canales.extend(train)
    test_canales.extend(test)

print(test_canales)
input("Mira")

# dataset['Conjunto'] = dataset['idcanal'].apply(lambda x: 'test' if x in test_canales else 'train')
dataset['Conjunto'] = dataset['idcanal'].apply(lambda x: 'train' if x in test_canales else 'test')
dataset = dataset.groupby(['Conjunto', 'Ideologia'], group_keys=False).apply(crear_clusters)

dataset.to_csv('videos_clusterized.csv', index=False)