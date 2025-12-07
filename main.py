import argparse
from src.preprocessing import convert_videos_to_audio
from src.features import FeatureEngine
from src.trainer import train_model

def main():
    """
    Main entry point for the Digital Soul pipeline.
    
    Parses command line arguments to execute specific stages of the project:
    - 'prep': Converts raw video to standardized audio.
    - 'extract': runs Whisper/Librosa to create feature vectors.
    - 'train': runs the training loop.
    - 'all': runs the entire pipeline sequentially.
    """
    parser = argparse.ArgumentParser(
        description="Digital Soul: Personality Prediction Pipeline",
        epilog="Example: 'uv run main.py train' to start training."
    )
    parser.add_argument(
        'mode', 
        choices=['prep', 'extract', 'train', 'all'], 
        help="The pipeline stage to execute."
    )
    
    args = parser.parse_args()
    
    print(f"Executing mode: {args.mode.upper()}")

    if args.mode in ['prep', 'all']:
        convert_videos_to_audio()
        
    if args.mode in ['extract', 'all']:
        engine = FeatureEngine()
        engine.run_batch()

    if args.mode in ['train', 'all']:
        train_model()

if __name__ == "__main__":
    main()