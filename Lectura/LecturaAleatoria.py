import pandas as pd
import random

df = pd.read_csv('videos.csv')
df = df.dropna(subset=['Transcripcion', 'Ideologia', 'idcanal'])

# Crear lista con tuplas: (idcanal, transcripcion, ideologia)
lista_datos = list(df[['idcanal', 'Transcripcion', 'Ideologia']].itertuples(index=False, name=None))

print("Pulsa ENTER para ver una transcripción aleatoria, luego ENTER para ver su ideología e idcanal. Escribe 'salir' para terminar.")

while True:
    entrada = input("Presiona ENTER para nueva transcripción o escribe 'salir': ")
    if entrada.lower() == 'salir':
        print("Saliendo...")
        break
    
    idcanal, transcripcion, ideologia = random.choice(lista_datos)
    print("\n--- Transcripción ---")
    print(transcripcion)
    
    input("\nIntenta adivinar la ideología y pulsa ENTER para verla junto con el idcanal...")
    print(f"Ideología: {ideologia}")
    print(f"idcanal: {idcanal}")
    print("------------------------------")
