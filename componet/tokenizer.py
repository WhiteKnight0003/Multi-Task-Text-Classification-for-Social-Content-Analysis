from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from componet.load_data import DataLoader
from componet.preprocessor import DataPreprocessor
from componet.split import DataSplitter
import warnings
warnings.filterwarnings('ignore')

class TextTokenizer:
    """Tokenize and pad text sequences"""
    def __init__(self, max_length=50):
        self.max_length = max_length
        self.tokenizer = Tokenizer()

    def fit_tokenizer(self, x_emotion_train, x_violence_train, x_hate_train):
        """Fit tokenizer on all training data"""
        all_texts = pd.concat([x_emotion_train, x_violence_train, x_hate_train])
        self.tokenizer.fit_on_texts(all_texts)
        print(f"Vocabulary size: {len(self.tokenizer.word_index) + 1}")
    

    def tokenize_and_pad(self, texts):
        """Convert texts to padded sequences"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        return padded
    
    
    def process_splits(self, train_test_splits):
        """Process all train/test splits"""
        processed = {}
        
        for task, (x_train, x_test, y_train, y_test) in train_test_splits.items():
            x_train_padded = self.tokenize_and_pad(x_train)
            x_test_padded = self.tokenize_and_pad(x_test)
            y_train_labels = np.array(y_train)
            y_test_labels = np.array(y_test)
            
            processed[task] = {
                'x_train': x_train_padded,
                'x_test': x_test_padded,
                'y_train': y_train_labels,
                'y_test': y_test_labels
            }
        
        return processed

if __name__ == "__main__":
    # Define paths
    emotion_path = './data/Emotions/text.csv'
    violence_path = './data/Gender-Based_Violence_Tweet_Classification/train.csv'
    hate_path = './data/Hate_Speech_and_Offensive_Language_Dataset/labeled_data.csv'

    data = DataLoader(emotion_path, violence_path, hate_path)
    emotion_df, violence_df, hate_df = data.load_data()

    preprocessor = DataPreprocessor()
    emotion_df, violence_df, hate_df = preprocessor.process(
        emotion_df, violence_df, hate_df
    )
    
    splitter = DataSplitter()
    train_test_splits = splitter.split(emotion_df, violence_df, hate_df)

    tokenizer = TextTokenizer()
    tokenizer.fit_tokenizer(
        train_test_splits['emotion'][0],
        train_test_splits['violence'][0],
        train_test_splits['hate'][0]
    )
    processed_data = tokenizer.process_splits(train_test_splits)
    print(processed_data)
    