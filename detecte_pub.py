import os
import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment
from tensorflow.keras.models import load_model
import json
from difflib import SequenceMatcher
from datetime import datetime
import whisper
import warnings
import torch

# Supprimer les avertissements liés à torch.load et FP16
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Paramètres améliorés
sample_rate = 22050
target_shape = (40, 86)
window_size = 5000  
overlap = 2500  
threshold = 0.5  
confidence_threshold = 0.7  
min_ad_duration = 10 * 1000  

# Extraction MFCCs
def extract_mfcc_features(audio, sr):
    audio = nr.reduce_noise(y=audio, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=target_shape[0])
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    mfccs = librosa.util.fix_length(mfccs, size=target_shape[1], axis=1)
    return mfccs.reshape(1, target_shape[0], target_shape[1], 1)

# Classification audio avec seuil de confiance
def classify_audio(segment_audio, model):
    features = extract_mfcc_features(segment_audio, sample_rate)
    
    prob = model.predict(features)[0][0]  # Prédiction unique
    
    label = "Publicité" if prob > threshold and prob >= confidence_threshold else "Non-Publicité"
    
    print(f"Prédiction: {label}, Probabilité: {prob:.2f}")
    return label, prob

# Fonction pour transcrire un segment audio avec Whisper
def transcribe_audio(segment_audio):
    temp_file_path = "temp_segment.wav"
    
    # Exporter le segment audio en fichier WAV
    try:
        segment_audio.export(temp_file_path, format="wav")
        if not os.path.exists(temp_file_path):
            raise FileNotFoundError(f"Le fichier {temp_file_path} n'a pas été créé.")
    except Exception as e:
        print(f"Erreur lors de l'exportation du segment audio : {str(e)}")
        return ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Charger le modèle avancé de Whisper
    try:
        model = whisper.load_model("large-v3", device=device)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Whisper : {str(e)}")
        return ""

    # Transcription
    try:
        result = model.transcribe(temp_file_path, language="fr", fp16=torch.cuda.is_available())
    except Exception as e:
        print(f"Erreur lors de la transcription : {str(e)}")
        return ""
    
    # Suppression du fichier temporaire
    try:
        os.remove(temp_file_path)
    except Exception as e:
        print(f"Erreur lors de la suppression du fichier temporaire : {str(e)}")

    return result['text']

# Fonction pour sauvegarder une publicité détectée avec ses métadonnées dans un fichier JSON
def save_ad_segment(audio, start, end, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ad_segment = audio[start:end]
    output_file = os.path.join(output_dir, f"ad_{start // 1000}.wav")
    
    ad_segment.export(output_file, format="wav")
    
    duration = (end - start) / 1000  # Durée en secondes
    print(f"Publicité sauvegardée : {output_file}, Durée : {duration:.2f}s")

    # Transcription du segment audio
    transcription = transcribe_audio(ad_segment)

    # Sauvegarder les métadonnées dans un fichier JSON
    metadata = {
        "start": str(datetime.fromtimestamp(start / 1000)),  
        "end": str(datetime.fromtimestamp(end / 1000)),
        "duration": duration,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transcription": transcription.strip()
    }
    
    json_file_path = os.path.join(output_dir, f"ad_{start // 1000}.json")
    
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(metadata, json_file, ensure_ascii=False, indent=4)
    
    print(f"Métadonnées sauvegardées : {json_file_path}")

def are_similar(text1, text2, threshold=0.6):
    """Compare deux textes et retourne True s'ils sont similaires."""
    similarity = SequenceMatcher(None, text1, text2).ratio()
    return similarity >= threshold   

# Détection et fusion des publicités
def detect_ads(audio_file, model, output_dir="ads_output"):
    print(f"Analyse de {audio_file}...")
    audio = AudioSegment.from_wav(audio_file)

    in_ad = False
    ad_start = None

    for start in range(0, len(audio) - window_size + overlap, overlap):
        end = start + window_size
        
        if end > len(audio):
            break
        
        segment = audio[start:end]
        y = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0
        
        label, prob = classify_audio(y, model)
        
        if label == "Publicité":
            if not in_ad:  # Si nous entrons dans une publicité
                ad_start = start
                in_ad = True
                
            # Si nous sommes déjà dans une publicité et que nous détectons à nouveau une publicité,
            # nous ne faisons rien car nous voulons fusionner les segments.
            
        else:  # Si nous sortons d'une publicité
            if in_ad:
                ad_end = start
                
                # Vérifier si la durée est suffisante avant la sauvegarde
                duration = ad_end - ad_start
                
                if duration >= min_ad_duration:
                    save_ad_segment(audio, ad_start, ad_end, output_dir)
                else:
                    print(f"Publicité rejetée : Durée trop courte ({duration / 1000:.2f}s)")
                
                in_ad = False

            # Si on sort d'une pub mais qu'on est proche d'une autre pub potentielle,
            # on peut ajuster le comportement ici si nécessaire.

    # Si la dernière publicité se termine à la fin de l'audio
    if in_ad:
        ad_end = len(audio)
        duration = ad_end - ad_start
        
        if duration >= min_ad_duration:
            save_ad_segment(audio, ad_start, ad_end, output_dir)

# Fonction principale 
def main(input_file, model_path, output_dir):
    model = load_model(model_path)
    
    detect_ads(input_file, model, output_dir)

# Exécution avec un répertoire de sortie spécifié 
if __name__ == "__main__":
    test_file = "/Users/daoudaouedraogo/Desktop/Medias/media/cont_pub/06_24_11_24_56.wav"
    keras_model_path = "/Users/daoudaouedraogo/Desktop/Medias/meilleur_modele2.keras"
    output_directory = "/Users/daoudaouedraogo/Desktop/Medias/Detected_Ads1"  
    
    main(test_file, keras_model_path, output_directory)


