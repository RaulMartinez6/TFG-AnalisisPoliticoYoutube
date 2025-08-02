# Creación de perfiles de usario a partir de contenido textual de Youtube

**TFG - Grado en Ingeniería Informática – Universidad de Murcia**  
**Autor:** Raúl Martínez Campos  
**Tutor:** Rafael Valencia García  
**Cotutor:** Ronghao Pan
**Fecha:** Julio de 2025

Este trabajo aborda la detección automática de la ideología política en emisores digitales mediante el análisis de transcripciones de vídeos publicados en YouTube. Para ello, se ha construido un corpus original compuesto por más de 19.000 vídeos procedentes de 21 canales distintos, clasificados según su orientación ideológica en cuatro categorías: izquierda, izquierda moderada, derecha moderada y derecha. A partir de este conjunto de datos, se agruparon vídeos en clústeres ideológicos y se entrenaron diversos modelos de lenguaje preentrenados —como BETO, RoBERTa-BNE, DistilBERT y mBERT— para realizar la tarea de clasificación.

Los experimentos realizados revelan un alto rendimiento en escenarios donde los datos de entrenamiento y prueba comparten canales, con valores de precisión y macro F1 superiores al 0.98. Sin embargo, al introducir una separación estricta por canal entre los conjuntos, el rendimiento cae por debajo del azar, lo que evidencia una fuerte dependencia de los modelos respecto a señales estilísticas o léxicas específicas de cada emisor. Esta hipótesis se refuerza mediante una evaluación externa con el corpus anotado PoliticES, que confirma el correcto funcionamiento de los modelos en un entorno controlado y con anotaciones más coherentes a nivel discursivo.

Además del análisis cuantitativo, se ha llevado a cabo una exploración cualitativa del corpus, que ha puesto de manifiesto varios factores que dificultan la clasificación, como la ambigüedad ideológica de algunos vídeos, la inclusión de contenido neutro o poco informativo, y la inconsistencia en la asignación ideológica en algunos casos, al haberse realizado a nivel de canal en lugar de por vídeo. En consecuencia, aunque las arquitecturas utilizadas demuestran un gran potencial técnico, se concluye que el rendimiento práctico en tareas como esta depende en gran medida de la calidad, definición y consistencia del corpus. Como línea de mejora prioritaria, se propone refinar el proceso de anotación mediante la asignación de etiquetas ideológicas a nivel individual de vídeo, lo que permitiría reducir la ambigüedad y mejorar la capacidad de generalización de los sistemas desarrollados.
