import pandas as pd
from sklearn.model_selection import train_test_split
import random

tam_cluster = 5

#Divide en cluster los videos de una ideologia
def crear_clusters(df_test, ideologia):
    subset = df_test[df_test['Ideologia'] == ideologia].copy()

    # Mezcla los videos segun el canal
    canales = subset['idcanal'].unique()
    canal_videos = [subset[subset['idcanal'] == canal].sample(frac=1, random_state=42).reset_index(drop=True)
                    for canal in canales]

    intercalado = []
    while any(len(cv) > 0 for cv in canal_videos):
        for i in range(len(canal_videos)):
            if len(canal_videos[i]) > 0:
                intercalado.append(canal_videos[i].iloc[0])
                canal_videos[i] = canal_videos[i].iloc[1:]

    intercalado_df = pd.DataFrame(intercalado).reset_index(drop=True)

    # Crear clusters
    clusters = []
    cluster_number = 1
    i = 0
    while i < len(intercalado_df):
        remaining = len(intercalado_df) - i
        # Control de de tamaño del cluster
        if remaining < tam_cluster // 2 and clusters:
            clusters[-1] = pd.concat([clusters[-1], intercalado_df.iloc[i:]], ignore_index=True)
            break
        else:
            cluster = intercalado_df.iloc[i:i+tam_cluster].copy()
            cluster['Cluster'] = cluster_number
            clusters.append(cluster)
            cluster_number += 1
            i += tam_cluster

    return pd.concat(clusters, ignore_index=True)

dataset = pd.read_csv('videos.csv')

#Dividir en train/test
conjuntos = []
for canal_id, grupo in dataset.groupby("idcanal"):
    grupo = grupo.sample(frac=1, random_state=42).reset_index(drop=True)
    grupo_train, grupo_test = train_test_split(grupo, test_size=0.2, random_state=42, shuffle=False)

    grupo_train['Conjunto'] = 'train'
    grupo_test['Conjunto'] = 'test'

    conjuntos.append(grupo_train)
    conjuntos.append(grupo_test)

dataset = pd.concat(conjuntos, ignore_index=True)

#Separar train y test
test_df = dataset[dataset['Conjunto'] == 'test'].copy()
train_df = dataset[dataset['Conjunto'] == 'train'].copy()

# Crear clusters para test balanceados por ideología
clustered_test = []
for ideologia in test_df['Ideologia'].unique():
    clustered = crear_clusters(test_df, ideologia)
    clustered_test.append(clustered)


final_df = pd.concat([train_df] + clustered_test).reset_index(drop=True)
final_df.to_csv('videos_clusterized.csv', index=False)
