import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

# Descargar stopwords de nltk si no las tienes
nltk.download('stopwords')

# Cargar CSV
df = pd.read_csv('videos.csv')  # Cambia el nombre del archivo

# Eliminar NaNs en Transcripcion e Ideologia
# df = df.dropna(subset=['Transcripcion', 'Ideologia'])
df = df.dropna(subset=['Titulo', 'Transcripcion', 'Ideologia'])

# Crear nueva columna combinando Titulo y Transcripcion
df['Texto_Completo'] = df['Titulo'] + ' ' + df['Transcripcion']


# Obtener lista completa de stopwords en español desde nltk
stopwords_es = stopwords.words('spanish')

# Agrupar por ideología
ideologias = df['Ideologia'].unique()

# Inicializar CountVectorizer con stopwords en español
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words=stopwords_es)

# Resultado final
resultados = {}

for ideologia in ideologias:
    subset = df[df['Ideologia'] == ideologia]
    # X = vectorizer.fit_transform(subset['Transcripcion'])
    X = vectorizer.fit_transform(subset['Texto_Completo'])
    suma_palabras = X.sum(axis=0)
    
    # Crear dataframe de frecuencia
    palabras_freq = pd.DataFrame({
        'palabra': vectorizer.get_feature_names_out(),
        'frecuencia': suma_palabras.A1
    }).sort_values(by='frecuencia', ascending=False)
    
    resultados[ideologia] = palabras_freq

    print(f"\n--- Palabras más frecuentes para ideología: {ideologia} ---")
    print(palabras_freq.to_string(index=False))
    filename = f'frecuencias_ideologia_{ideologia}.csv'
    palabras_freq.to_csv(filename, index=False)
    print(f"Guardado archivo: {filename}")