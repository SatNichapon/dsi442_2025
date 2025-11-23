import os
from moviepy import VideoFileClip

# --- SETTINGS ---
VIDEO_DIR = "data/raw_videos"      # Where your mp4s are
AUDIO_DIR = "data/processed_audio" # Where wavs will go
LIMIT = 50                         # <--- THE HACK: Only process 50 files for now

# Ensure output folder exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# Get list of video files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

print(f"Found {len(video_files)} videos. Processing the first {LIMIT}...")

count = 0
for video_file in video_files:
    if count >= LIMIT:
        break # Stop after 50
        
    try:
        # Define paths
        video_path = os.path.join(VIDEO_DIR, video_file)
        audio_filename = video_file.replace(".mp4", ".wav")
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        # Check if already exists (skip if yes)
        if os.path.exists(audio_path):
            print(f"Skipping {audio_filename} (already exists)")
            count += 1
            continue

        # Extract Audio
        clip = VideoFileClip(video_path)
        # Write audio: 16kHz, mono channel (best for Whisper/Librosa)
        clip.audio.write_audiofile(audio_path, logger=None, fps=16000, nbytes=2, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
        
        print(f"[{count+1}/{LIMIT}] Converted: {audio_filename}")
        count += 1
        
    except Exception as e:
        print(f"❌ Error processing {video_file}: {e}")

print("✅ Micro-Batch Audio Extraction Complete!")