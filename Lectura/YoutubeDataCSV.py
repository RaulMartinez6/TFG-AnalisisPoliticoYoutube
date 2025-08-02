import argparse
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, YouTubeRequestFailed
import csv
import os
import re
import xml

archivo_csv = "videosPrueba.csv"
api_service_name = "youtube"
api_version = "v3"
api_key = "AIzaSyBKKl9Nxy_0O9FWNxeWJnKdBSOCjsjzXn0"

# Lee el fichero con nombres de canales pasado como paramatro y obtiene la informacion de todos sus videos
def get_videos_info(channelsFile):
    with open(channelsFile, 'r', encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
     
    videos = []
    for line in lines:
        channel_info = line.split()
        channel_id = channel_info[0]
        if len(channel_info) == 3:
            channel_ideology = channel_info[1] + " " + channel_info[2]
        else:
            channel_ideology = channel_info[1]

        # Obtener informacion de todos los videos del canal
        videos = get_videos_info_channel(channel_id, channel_ideology)
        guardar_en_csv(videos)

# Devuelve toda la informacion de los videos de channel_id
def get_videos_info_channel(channel_id, channel_ideology):
    # Crea un cliente con la API key
    youtube = build(api_service_name, api_version, developerKey=api_key)
    
    # Obtener videos de un canal
    if channel_id[0] == '@':
        request_channel = youtube.channels().list(
            part="contentDetails",
            forHandle=channel_id
        )
        response_channel = request_channel.execute()
        channel_url = channel_id
    else:
        request_channel = youtube.channels().list(
            part="snippet,contentDetails",
            id=channel_id
        )
        response_channel = request_channel.execute()
        channel_url = response_channel["items"][0]["snippet"].get("customUrl", "No disponible")
        
    print("Obteniendo los videos del canal " + channel_url)
    
    # Obtener lista de reproduccion de todos los videos subidos
    uploads_playlist_id = response_channel["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    max_videos = 1000
    max_transcripcion = 8000
    videos = []

    # Obtener videos de una playList
    request_videos = youtube.playlistItems().list(
        part="snippet",
        playlistId=uploads_playlist_id,
        maxResults=50
    )
    num_rechazos = 0
    while request_videos and len(videos) < max_videos:
        try:
            response_videos = request_videos.execute()
            for item in response_videos["items"]:

                video_id = item["snippet"]["resourceId"]["videoId"]
                video_title = item["snippet"]["title"]
                video_description = item["snippet"]["description"]
                video_published = item["snippet"]["publishedAt"]

                # Obtener la transcipcion si se puede
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es'])
                    transcripcion = " ".join([entry["text"] for entry in transcript])
                    
                    # Eliminar secciones donde solo se escucha musica de fondo y restringuir el tamaño de la cadena
                    transcripcion = re.sub(r'\[.*?\]', '', transcripcion)
                    if len(transcripcion) > max_transcripcion:
                        transcripcion = transcripcion[:max_transcripcion].rsplit(' ', 1)[0]

                    # Peticion necesaria para conseguir los hastatags 
                    request_stats = youtube.videos().list(
                        part="statistics,snippet",
                        id=video_id
                    )
                    response_stats = request_stats.execute()
                    video_tags = response_stats["items"][0]["snippet"].get("tags", [])
                    
                    # Agregar información al listado
                    videos.append({
                        "idcanal": channel_url,
                        "idvideo": video_id,
                        "Titulo": video_title,
                        "Descripcion": video_description,
                        "Fecha": video_published,
                        "Hashtags": ", ".join(video_tags),
                        "Transcripcion": transcripcion,
                        "Ideologia": channel_ideology
                    })
                    
                    print(f"Video {len(videos)}: {video_title}")
                except (TranscriptsDisabled, NoTranscriptFound, YouTubeRequestFailed, xml.etree.ElementTree.ParseError):
                    num_rechazos += 1

                # Continuar si hay más páginas
            request_videos = youtube.playlistItems().list_next(
                request_videos, response_videos
            )
        except Exception as e:
            if hasattr(e, 'resp') and hasattr(e.resp, 'status') and e.resp.status == 403:  # Límite de cuota alcanzado
                print("¡Limite de cuota de la API alcanzado. Guardando los videos obtenidos hasta el momento...")
                guardar_en_csv(videos)  # Guardar los resultados hasta el momento
                print(f"Se han procesado {len(videos)} videos de {channel_url}.")
                break  # Detener el proceso si se alcanza el límite
            else:
                print(f"Error inesperado: {e}")
                break  # Detener el proceso ante un error inesperado
    
    print(f"No se han encontrado {num_rechazos} transcripciones")
    print(f"Se han procesado {len(videos)} videos de {channel_url}.")
    return videos

def guardar_en_csv(videos):
    file_exists = os.path.exists(archivo_csv)
    with open(archivo_csv, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["idcanal", "idvideo", "Titulo", "Descripcion", "Fecha", "Hashtags", "Transcripcion","Ideologia"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(videos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar un archivo de canales")
    parser.add_argument('file', type=str, help="El archivo que contiene los IDs de los canales")
    args = parser.parse_args()
    file = args.file

    get_videos_info(file)