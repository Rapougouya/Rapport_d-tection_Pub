import os
import numpy as np
import librosa
from pydub import AudioSegment, silence
import tensorflow as tf
import noisereduce as nr
from datetime import datetime, timedelta
import whisper
import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Désactiver les warnings
import warnings
warnings.filterwarnings("ignore")

# Paramètres
sample_rate = 22050
max_pad_len = 86
output_dir_pub = '/Users/daoudaouedraogo/Desktop/Medias/audios_segmentes/publicites'
confidence_threshold = 0.9  # Seuil de confiance pour publicité
os.makedirs(output_dir_pub, exist_ok=True)

# Fonction pour convertir un fichier audio en WAV
def convert_to_wav(file_path):
    print(f"Conversion de {file_path} en WAV")
    if file_path.endswith('.mp3'):
        sound = AudioSegment.from_mp3(file_path)
        wav_path = file_path.replace('.mp3', '.wav')
        sound.export(wav_path, format="wav")
        return wav_path
    return file_path

# Fonction pour extraire des caractéristiques MFCC
def extract_mfcc_features(file_path, target_shape=(40, 86)):
    print(f"Extraction des caractéristiques MFCC pour {file_path}")
    audio, sr = librosa.load(file_path, sr=sample_rate)
    audio = nr.reduce_noise(y=audio, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=target_shape[0])
    # Normalisation des MFCC
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    # Ajustement à la forme cible
    if mfccs.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > target_shape[1]:
        mfccs = mfccs[:, :target_shape[1]]
    return mfccs

# Fonction pour prédire la classe d'un segment
def test_segment(segment_path, model):
    try:
        input_features = extract_mfcc_features(segment_path)
        input_features = np.expand_dims(input_features, axis=-1)
        input_features = np.expand_dims(input_features, axis=0)
        prediction = model.predict(input_features)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        predicted_confidence = float(np.max(prediction, axis=1)[0])
        return predicted_class, predicted_confidence
    except Exception as e:
        print(f"Erreur pour le segment {segment_path} : {e}")
        return None, None

# Fonction pour transcrire un fichier audio
def transcribe_audio(file_path):
    try:
        print(f"Transcription de {file_path}")
        whisper_model = whisper.load_model("large-v2", download_root=None, in_memory=False)
        result = whisper_model.transcribe(file_path)
        transcription = result.get("text", "")
        return transcription
    except Exception as e:
        print(f"Erreur lors de la transcription de {file_path} : {e}")
        return ""

# Fonction pour segmenter un fichier audio en fonction des silences
def segment_by_silence(file_path, silence_thresh=-45, min_silence_len=500, segment_duration=60, overlap=10):
    print(f"Segmentation par silence de {file_path}")
    audio = AudioSegment.from_wav(file_path)
    segments = []
    silent_intervals = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    silent_intervals = [[0, 0]] + silent_intervals + [[len(audio), len(audio)]]
    for i in range(len(silent_intervals) - 1):
        start = silent_intervals[i][1]
        end = silent_intervals[i + 1][0]
        if end - start > segment_duration * 1000:
            for j in range(start, end, segment_duration * 1000 - overlap * 1000):
                segment = audio[j:j + segment_duration * 1000]
                segment_path = os.path.join(output_dir_pub, f"segment_{j // 1000}_{(j + len(segment)) // 1000}.wav")
                segment.export(segment_path, format="wav")
                segments.append((segment_path, j, j + len(segment)))
        else:
            segment = audio[start:end]
            segment_path = os.path.join(output_dir_pub, f"segment_{start // 1000}_{end // 1000}.wav")
            segment.export(segment_path, format="wav")
            segments.append((segment_path, start, end))
    return segments

# Fonction pour traiter les segments de publicités détectées
def process_segments(segments, model):
    for segment_path, start_ms, end_ms in segments:
        if os.path.exists(segment_path):
            predicted_class, predicted_confidence = test_segment(segment_path, model)
            if predicted_class == 1 and predicted_confidence >= confidence_threshold:
                transcription = transcribe_audio(segment_path)
                start_time = timedelta(milliseconds=start_ms)
                end_time = timedelta(milliseconds=end_ms)
                duration = end_time - start_time
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                segment_name = os.path.splitext(os.path.basename(segment_path))[0]
                json_path = os.path.join(output_dir_pub, f"{segment_name}.json")
                # Sauvegarder les informations dans un fichier JSON
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "start": str(start_time),
                        "end": str(end_time),
                        "duration": str(duration),
                        "transcription": transcription,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=4, ensure_ascii=False)
                print(f"Publicité sauvegardée : {segment_path}, Transcription dans {json_path}")

# Surveillance des fichiers
class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith((".mp3", ".wav")):
            process_file(event.src_path)

def watch_directory(directory):
    observer = Observer()
    observer.schedule(FileHandler(), directory, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Traitement principal
def process_file(file_path):
    try:
        print(f"Traitement du fichier : {file_path}")
        wav_file_path = convert_to_wav(file_path)
        segments = segment_by_silence(wav_file_path)
        process_segments(segments, model)
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {file_path} : {e}")

if __name__ == "__main__":
    model_path = '/Users/daoudaouedraogo/Desktop/Medias/model_uniform.keras'
    try:
        # Charger le modèle
        model = tf.keras.models.load_model(model_path)
        # Démarrer la surveillance
        watch_directory('/Users/daoudaouedraogo/Desktop/Medias/Audios')
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")