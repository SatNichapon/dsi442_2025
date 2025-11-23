import pandas as pd
import pickle
import os

LABEL_PATH = "data/ground_truth/train-annotation/annotation_training.pkl" 

if not os.path.exists(LABEL_PATH):
    print("File not found. Check the path!")
else:
    try:
        # Try loading as Pickle (Common for ChaLearn)
        with open(LABEL_PATH, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        print("Success! File opened.")
        
        if isinstance(data, dict):
            # Convert dictionary to DataFrame
            df = pd.DataFrame.from_dict(data, orient='index')
            
            # Reset index so 'filename' becomes a column
            df.index.name = 'video_name'
            df.reset_index(inplace=True)
            
            print(f"📊 Found scores for {len(df)} videos.")
            print("\n--- First 5 Rows ---")
            print(df.head())
            
            # CHECK: Do the filenames match your processed audio?
            first_video = df.iloc[0]['video_name']
            print(f"\nExample Label: {first_video}")
            print(f"Example Audio: {first_video.replace('.mp4', '.wav')}")
            
        else:
            print("Data format is not a dictionary. It looks like:", type(data))

    except Exception as e:
        print(f"Error reading pickle: {e}")