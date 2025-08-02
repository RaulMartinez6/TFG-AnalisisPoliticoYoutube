import torch
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import pandas as pd
import numpy as np
import argparse
from cf_matrix import make_confusion_matrix
import os
import matplotlib.pyplot as plt

label_mapping = {
    "right": 0,
    "moderate_right": 1,
    "moderate_left": 2,
    "left": 3
}

# Añade el respetivo label correspondiente a cada 'Ideologia'
def encode_labels(video):
    video['label'] = label_mapping[video['ideology_multiclass']]
    return video

# Concatena los textos de un video
def concatenate(video):
    tweet = video.get('tweet', "")
    video['input_text'] = f'{tweet}'
    return video

# Tokeniza el dataset
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['input_text'], padding="max_length", truncation=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ejecutar el modelo entrenado")
    parser.add_argument('model', type=str, help="Modelo guardado")
    args = parser.parse_args()
    trainer_dir = args.model

    # Prepara el csv para su uso
    test_dataset = load_dataset("csv", data_files="politicES_phase_2_test_codalab.csv")['train']
    test_dataset = test_dataset.add_column("Cluster", test_dataset['label'])
    test_dataset = test_dataset.map(concatenate)
    test_dataset = test_dataset.map(encode_labels)

    model = AutoModelForSequenceClassification.from_pretrained(trainer_dir, num_labels=4, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(trainer_dir)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Configuracion de argumentos de entrenamiento
    trainer = Trainer(model=model, tokenizer=tokenizer)
    
    # #Obtencion de metricas
    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(-1)
    true_labels = test_dataset['label']

    print(classification_report(true_labels,pred_labels,target_names=list(label_mapping.keys()), zero_division=0))

    test_dataset = test_dataset.add_column("predicted", pred_labels)

    test_dataset = test_dataset.to_pandas()
    cluster_results = test_dataset.groupby(['Ideologia', 'Cluster'])['predicted'].agg(lambda x: x.mode().iloc[0]).reset_index()
    cluster_results = cluster_results.apply(encode_labels, axis=1)

    # Mostrar el reporte de clasificación 
    print("Reporte de clasificación por Cluster:")
    print(classification_report(cluster_results['label'], cluster_results['predicted'], target_names=list(label_mapping.keys()), zero_division=0))
    
    # with open(os.path.join(trainer_dir, "classification_report.txt"), "a", encoding="utf-8") as f:
    #     f.write("\n\nReporte de clasificación por Cluster tam 5:\n")
    #     f.write(classification_report(cluster_results['label'], cluster_results['predicted'], target_names=list(label_mapping.keys()), zero_division=0))
    
    # Crear la matriz de confusión usando las predicciones por clusters
    conf_matrix = confusion_matrix(cluster_results['label'], cluster_results['predicted'])
    make_confusion_matrix(conf_matrix,
                        categories=list(label_mapping.keys()),
                        title=f"Matriz de Confusión - {trainer_dir}")
    plt.savefig(os.path.join(trainer_dir, "matriz_confusion.png"), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.show()