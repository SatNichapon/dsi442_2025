import os
from moviepy import VideoFileClip

# --- SETTINGS ---
VIDEO_ROOT = "data/raw_videos/val-1"    
AUDIO_DIR = "data/processed_audio" 
LIMIT = 50                         

# Ensure output folder exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# Find all MP4 files recursively
all_video_paths = []
for root, dirs, files in os.walk(VIDEO_ROOT):
    for file in files:
        if file.endswith(".mp4"):
            full_path = os.path.join(root, file)
            all_video_paths.append(full_path)

print(f"Found {len(all_video_paths)} videos total.")
print(f"Processing the first {LIMIT}...")

# Process the audio
count = 0
for video_path in all_video_paths:
    if count >= LIMIT:
        break
        
    try:
        # Get the filename
        filename = os.path.basename(video_path)
        
        # Create output filename
        audio_filename = filename.replace(".mp4", ".wav")
        audio_save_path = os.path.join(AUDIO_DIR, audio_filename)
        
        # Check if already exists
        if os.path.exists(audio_save_path):
            print(f"Skipping {audio_filename} (already exists)")
            count += 1
            continue

        # Extract Audio
        clip = VideoFileClip(video_path)
        
        # Write audio: 16kHz, mono channel
        clip.audio.write_audiofile(
            audio_save_path, 
            logger=None, 
            fps=16000, 
            nbytes=2, 
            codec='pcm_s16le', 
            ffmpeg_params=["-ac", "1"]
        )
        
        # Close clip to free memory
        clip.close()
        
        print(f"[{count+1}/{LIMIT}] Converted: {audio_filename}")
        count += 1
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

print("Micro-Batch Audio Extraction Complete!")