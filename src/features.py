import os
import numpy as np
import librosa
import torch
import whisper
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import config

class FeatureEngine:
    def __init__(self):
        self.device = config.DEVICE
        print(f"Initializing Feature Engine on {self.device}...")
        
        # Load Models
        # Whisper
        self.whisper = whisper.load_model("base", device=self.device)
        
        # ModernBERT
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.text_model = AutoModel.from_pretrained(config.MODEL_NAME).to(self.device)
        self.text_model.eval()

    def process_file(self, audio_path, filename):
        """
        Extracts and saves linguistic and acoustic features for a single audio file.

        This method processes the input audio through two parallel streams:
        1. Acoustic: Uses Librosa to extract MFCCs (saved as *_acou.npy).
        2. Linguistic: Uses Whisper for ASR and RoBERTa for text embeddings (saved as *_ling.npy).

        Args:
            audio_path (str): The full file path to the .wav audio file.
            filename (str): The basename of the file (e.g., "video123.wav"), used for naming output files.

        Returns:
            None: This method saves files to disk directly and does not return values.
        """
        ling_path = os.path.join(config.FEATURES_DIR, filename.replace(".wav", "_ling.npy"))
        acou_path = os.path.join(config.FEATURES_DIR, filename.replace(".wav", "_acou.npy"))

        if os.path.exists(ling_path) and os.path.exists(acou_path):
            return # Skip

        # --- ACOUSTIC (Librosa) ---
        # Load max 15s to keep sizes manageable
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_Rate, duration=config.MAX_DURATION)
        # Pad if too short (critical for LSTM)
        target_len = config.SAMPLE_Rate * config.MAX_DURATION
        if len(y) < target_len:
            y = librosa.util.fix_length(y, size=target_len)
        else:
            y = y[:target_len]
            
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        np.save(acou_path, mfcc)

        # --- LINGUISTIC (Whisper + RoBERTa) ---
        # 1. Transcribe
        res = self.whisper.transcribe(audio_path)
        text = res['text'].strip() or "silence"
        
        # 2. Embed
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=config.MAX_TEXT_LEN
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)

        # ModernBERT's [CLS] token is usually at index 0.
        last_hidden_state = outputs.last_hidden_state
        
        # Take the embedding of the [CLS] token (index 0)
        vec = last_hidden_state[:, 0, :].cpu().numpy().squeeze()

        # Save [CLS] pooling vector
        np.save(ling_path, vec)

    def run_batch(self):
        print("Starting Feature Extraction...")
        audio_files = [f for f in os.listdir(config.AUDIO_DIR) if f.endswith(".wav")]
        
        # Respect limit if needed
        if config.DATA_LIMIT:
            audio_files = audio_files[:config.DATA_LIMIT]

        for f in tqdm(audio_files):
            path = os.path.join(config.AUDIO_DIR, f)
            try:
                self.process_file(path, f)
            except Exception as e:
                print(f"Error extracting {f}: {e}")