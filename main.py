import pandas as pd
import numpy as np
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
from componet.pipeline import MultiTaskPipeline
import os

if __name__ == "__main__":
    # Define paths
    emotion_path = './data/Emotions/text.csv'
    violence_path = './data/Gender-Based_Violence_Tweet_Classification/train.csv'
    hate_path = './data/Hate_Speech_and_Offensive_Language_Dataset/labeled_data.csv'
    
    # Định nghĩa đường dẫn lưu model
    model_save_dir = './saved_models'
    model_save_path = os.path.join(model_save_dir, 'best_model.weights.h5')
    
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    MODE = 'train' # load
    # Run pipeline
    pipeline = MultiTaskPipeline(emotion_path, violence_path, hate_path)
    classifier = pipeline.run(epochs=10, batch_size=16 , model_save_path=model_save_path, mode=MODE)
    
    # Test classifier
    test_texts = [
        "I am so happy, this is the best day of my life!",
        "He grabbed me and punched me in the face.",
        "You are a stupid bitch and I hate you."
    ]
    
    print("\n" + "=" * 50)
    print("Testing Classifier")
    print("=" * 50)
    
    for text in test_texts:
        results = classifier.classify(text)
        print(f"\nInput: '{text}'")
        print("--- Predictions ---")
        print(f"Emotion:  {results['Emotion']}")
        print(f"Violence: {results['Violence']}")
        print(f"Hate:     {results['Hate']}")