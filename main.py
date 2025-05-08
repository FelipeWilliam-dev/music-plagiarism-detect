import librosa
import numpy as np
from scipy.spatial.distance import cdist
from librosa.sequence import dtw
from pydub import AudioSegment
import os

def convert_mp3_to_wav(mp3_path):
    if not mp3_path.lower().endswith('.mp3'):
        return mp3_path
    wav_path = mp3_path.replace('.mp3', '.wav')
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format='wav')
    return wav_path

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=60.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return mfcc, chroma

def calculate_similarity(A_mfcc, A_chroma, B_mfcc, B_chroma):
    C_mfcc = cdist(A_mfcc.T, B_mfcc.T, metric='cosine')
    C_chroma = cdist(A_chroma.T, B_chroma.T, metric='cosine')
    C_mfcc = np.where(np.isnan(C_mfcc), 1.0, C_mfcc)
    C_chroma = np.where(np.isnan(C_chroma), 1.0, C_chroma)

    _, wp_mfcc = dtw(C=C_mfcc)
    _, wp_chroma = dtw(C=C_chroma)

    dist_mfcc = np.mean([C_mfcc[i, j] for i, j in wp_mfcc])
    dist_chroma = np.mean([C_chroma[i, j] for i, j in wp_chroma])

    similarity = 1 - (0.6 * dist_mfcc + 0.4 * dist_chroma)
    return similarity

# Inputs
path_a = convert_mp3_to_wav("Music/Adele.mp3")
path_b = convert_mp3_to_wav("Music/Mulheres.mp3")

# Feature Extraction
a_mfcc, a_chroma = extract_features(path_a)
b_mfcc, b_chroma = extract_features(path_b)

# Similarity Calculation
sim = calculate_similarity(a_mfcc, a_chroma, b_mfcc, b_chroma)
print(f"Music A and B have a similarity score of: {sim:.2f}")
