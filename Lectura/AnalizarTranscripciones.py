import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_selection import mutual_info_classif
from nltk.corpus import stopwords

# Descargar stopwords de nltk si no las tienes
nltk.download('stopwords')

# Cargar CSV
df = pd.read_csv('videos.csv')  # Cambia el nombre del archivo

# Eliminar NaNs en Transcripcion e Ideologia
df = df.dropna(subset=['Titulo','Transcripcion', 'Ideologia'])

# Obtener lista completa de stopwords en español desde nltk
stopwords_es = stopwords.words('spanish')

# Agrupar por ideología
ideologias = df['Ideologia'].unique()

# Inicializar CountVectorizer con stopwords en español
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words=stopwords_es)
X_vec = vectorizer.fit_transform(df['Titulo'] + " " +df['Transcripcion'])

# Variable objetivo
Y = df['Ideologia']

# Calcular información mutua
mi_scores = mutual_info_classif(X_vec, Y, discrete_features=True)

palabras = vectorizer.get_feature_names_out()
mi_df = pd.DataFrame({'palabra': palabras, 'MI': mi_scores})
mi_df = mi_df.sort_values('MI', ascending=False).head(200)

# Cargar todos los CSV de frecuencias
archivos = {
    'Derecha': 'frecuencias_ideologia_Derecha.csv',
    'Derecha moderada': 'frecuencias_ideologia_Derecha moderada.csv',
    'Izquierda': 'frecuencias_ideologia_Izquierda.csv',
    'Izquierda moderada': 'frecuencias_ideologia_Izquierda moderada.csv'
}

# Leerlos y unirlos en una sola tabla
tabla_frecuencias = pd.DataFrame({'palabra': mi_df['palabra']})

for ideologia, archivo in archivos.items():
    df_freq = pd.read_csv(archivo)
    df_freq.columns = ['palabra', ideologia]  # Renombrar columna de frecuencia a ideología
    tabla_frecuencias = tabla_frecuencias.merge(df_freq, on='palabra', how='left')

# Rellenar posibles NaNs con 0
tabla_frecuencias = tabla_frecuencias.fillna(0)

# Añadir la columna de MI
tabla_frecuencias = tabla_frecuencias.merge(mi_df, on='palabra')

# Determinar ideología dominante por frecuencia
frecuencia_cols = list(archivos.keys())
tabla_frecuencias['Ideologia_dominante'] = tabla_frecuencias[frecuencia_cols].idxmax(axis=1)

# Ordenar por MI
tabla_frecuencias = tabla_frecuencias.sort_values('MI', ascending=False)

# Mostrar
print(tabla_frecuencias[['palabra', 'MI', 'Ideologia_dominante'] + frecuencia_cols].head(30))

tabla_str = tabla_frecuencias[['palabra', 'MI', 'Ideologia_dominante'] + frecuencia_cols].to_string(index=False)

# Guardar en archivo txt
with open('palabras_importantes_con_frecuencias.txt', 'w', encoding='utf-8') as f:
    f.write(tabla_str)