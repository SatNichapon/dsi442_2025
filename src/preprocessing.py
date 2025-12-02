import os
from moviepy import VideoFileClip
import config

def convert_videos_to_audio() -> None:
    """
    Batch converts MP4 video files to WAV audio files.

    This function recursively scans the directory specified in config.RAW_VIDEO_DIR
    for .mp4 files. It extracts the audio track, converts it to 16kHz mono WAV,
    and saves it to config.AUDIO_DIR.

    Behavior:
        - Skips files that have already been converted.
        - Respects config.DATA_LIMIT to process only a subset of files (useful for testing).
        - Prints progress every 10 files.
        - Catches and logs errors for individual files without crashing the entire batch.

    Returns:
        None: Files are written directly to disk.
    """
    print("Starting Video -> Audio Conversion...")
    
    # Gather files
    all_videos = []
    for root, dirs, files in os.walk(config.RAW_VIDEO_DIR):
        for file in files:
            if file.endswith(".mp4"):
                all_videos.append(os.path.join(root, file))
    
    limit = config.DATA_LIMIT if config.DATA_LIMIT else len(all_videos)
    print(f"Found {len(all_videos)} videos. Processing up to {limit}...")

    count = 0
    for i, video_path in enumerate(all_videos):
        if count >= limit:
            break

        filename = os.path.basename(video_path)
        audio_name = filename.replace(".mp4", ".wav")
        save_path = os.path.join(config.AUDIO_DIR, audio_name)

        if os.path.exists(save_path):
            count += 1
            continue # Skip existing
            
        try:
            clip = VideoFileClip(video_path)
            # Write audio: 16kHz, mono channel, PCM 16-bit
            clip.audio.write_audiofile(
                save_path,
                logger=None,
                fps=config.SAMPLE_Rate,
                nbytes=2,
                codec='pcm_s16le',
                ffmpeg_params=["-ac", "1"]
            )
            clip.close()
            count += 1
            if count % 10 == 0:
                print(f"   Processed {count}/{limit}")
                
        except Exception as e:
            print(f"Error on {filename}: {e}")

    print("Conversion Complete.")