import torch
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from cf_matrix import make_confusion_matrix

# Diccionario para mapear
label_mapping = {
    "Derecha": 0,
    "Derecha moderada": 1,
    "Izquierda moderada": 2,
    "Izquierda": 3
}

# Modelos usados
modelos = {
    "dccuchile": "dccuchile/bert-base-spanish-wwm-cased",
    "google-bert": "bert-base-cased",
    "distilbert": "distilbert-base-multilingual-cased",
    "roberta-bne": "PlanTL-GOB-ES/roberta-base-bne"
}

# Añade el respetivo label correspondiente a cada 'Ideologia'
def encode_labels(video):
    video['label'] = label_mapping[video['Ideologia']]
    return video

# Concatena los textos de un video
def concatenate(video):
    titulo = video.get('Titulo', "")
    transcripcion = video.get('Transcripcion', "")
    video['input_text'] = f"{titulo} {transcripcion}"
    return video

# Tokeniza el dataset
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['input_text'], padding="max_length", truncation=True)

if __name__ == "__main__":

    # Prepara el csv para su uso
    dataset = load_dataset("csv", data_files="videos_clusterized.csv")['train']
    dataset = dataset.map(concatenate)
    dataset = dataset.map(encode_labels)

    for nombre_modelo, modelo_path in modelos.items():
        print(f"\nEntrenando modelo: {nombre_modelo} ({modelo_path})")

        output_dir = f"./modelos/{nombre_modelo}"
        os.makedirs(output_dir, exist_ok=True)

        # Tokenizamos segun el modelo
        tokenizer = AutoTokenizer.from_pretrained(modelo_path)
        encoded_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

        train_dataset = encoded_dataset.filter(lambda x: x['Conjunto'] == "train")
        test_dataset = encoded_dataset.filter(lambda x: x['Conjunto'] == "test")

        model = AutoModelForSequenceClassification.from_pretrained(modelo_path, num_labels=4, torch_dtype="auto")

        # Configuracion de argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir="./test_trainer",
            eval_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size= 16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            save_strategy="no"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # #Obtencion de metricas
        
        predictions = trainer.predict(test_dataset)

        pred_labels = predictions.predictions.argmax(-1)
        true_labels = test_dataset['label']
        

        print("Sin Cluster:\n")
        print(classification_report(true_labels,pred_labels,target_names=list(label_mapping.keys()), zero_division=0))

        test_dataset = test_dataset.add_column("predicted", pred_labels)

        test_dataset = test_dataset.to_pandas()
        cluster_results = test_dataset.groupby(['Ideologia', 'Cluster'])['predicted'].agg(lambda x: x.mode().iloc[0]).reset_index()
        cluster_results = cluster_results.apply(encode_labels, axis=1)

        # Mostrar el reporte de clasificación 
        print("Reporte de clasificación por Cluster:")
        report = classification_report(
            cluster_results['label'], cluster_results['predicted'],
            target_names=list(label_mapping.keys()),
            zero_division=0,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(output_dir, "clasificacion_cluster.csv"))
        with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write("Sin Cluster:\n")
            f.write(classification_report(true_labels,pred_labels,target_names=list(label_mapping.keys()), zero_division=0))
            f.write("Reporte de clasificación por Cluster:\n")
            f.write(classification_report(cluster_results['label'], cluster_results['predicted'], target_names=list(label_mapping.keys()), zero_division=0))
    
        print(report_df)

        # Matriz de confusión
        conf_matrix = confusion_matrix(cluster_results['label'], cluster_results['predicted'])
        make_confusion_matrix(conf_matrix,
                        categories=list(label_mapping.keys()),
                        title=f"Matriz de Confusión - {nombre_modelo}")
        plt.savefig(os.path.join(output_dir, "matriz_confusion.png"), dpi=300, bbox_inches='tight')
        plt.clf()

    os.system("shutdown /s /t 0")