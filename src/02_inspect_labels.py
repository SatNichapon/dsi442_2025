import pandas as pd
import pickle
import os

# POINT THIS TO YOUR LABEL FILE
# It might be named 'annotation_training.pkl' or similar
LABEL_PATH = "data/ground_truth/annotation_training.pkl" 

if not os.path.exists(LABEL_PATH):
    print("❌ File not found. Check the path!")
else:
    try:
        # Try loading as Pickle (Common for ChaLearn)
        with open(LABEL_PATH, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        print("✅ Loaded as Pickle!")
        print(f"Type of data: {type(data)}")
        
        # Usually it's a Dictionary or DataFrame. Let's peek.
        if isinstance(data, dict):
            keys = list(data.keys())
            print(f"Keys: {keys[:5]}")
            print(f"First Item: {data[keys[0]]}")
        elif isinstance(data, pd.DataFrame):
            print(data.head())
            
    except Exception as e:
        print("Not a pickle file. Trying CSV...")
        try:
            df = pd.read_csv(LABEL_PATH)
            print("✅ Loaded as CSV!")
            print(df.head())
        except Exception as e2:
            print(f"❌ Could not read file: {e2}")