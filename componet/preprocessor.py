import nltk
from nltk.corpus import stopwords
import pandas as pd
from componet.load_data import DataLoader

class DataPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))

    def clean_dataframes(self, emotion_df, violence_df, hate_df):
        # Drop unnecessary columns
        emotion_df = emotion_df.drop(columns=['Unnamed: 0'], errors='ignore')
        violence_df = violence_df.drop(columns=['Tweet_ID'], errors='ignore')
        hate_df = hate_df[['tweet', 'class']]
        
        # Rename columns
        violence_df = violence_df.rename(columns={'tweet': 'text', 'type': 'label'})
        hate_df = hate_df.rename(columns={'tweet': 'text', 'class': 'label'})
        
        return emotion_df, violence_df, hate_df
    

    def balance_datasets(self, emotion_df, violence_df, hate_df):
        # Balance emotion dataset
        e_df = pd.DataFrame()
        for i in range(6):
            subset = emotion_df[emotion_df['label'] == i].sample(n=2000, random_state=42)
            e_df = pd.concat([e_df, subset])
        
        # Balance violence dataset
        sexual_violence = violence_df[violence_df['label'] == 'sexual_violence'].sample(n=4998, random_state=42)
        violence_df = violence_df[violence_df['label'] != 'sexual_violence']
        violence_df = pd.concat([sexual_violence, violence_df], axis=0)
        
        # Balance hate dataset
        offensive_speech = hate_df[hate_df['label'] == 1].sample(n=6407, random_state=42)
        hate_df = hate_df[hate_df['label'] != 1]
        hate_df = pd.concat([offensive_speech, hate_df], axis=0)
        
        return e_df, violence_df, hate_df
    
    def encode_labels(self, violence_df):
        
        violence_labels = ['sexual_violence', 'physical_violence', 
                               'emotional_violence','economic_violence', 'harmful_traditional_practice']
        violence_map = {label: i for i, label in enumerate(violence_labels)}
        violence_df['label'] = violence_df['label'].str.lower()
        violence_df['label'] = violence_df['label'].map(violence_map)
        violence_df['label'] = violence_df['label'].astype(int)
        
        return violence_df
    

    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        all_words = nltk.word_tokenize(text)
        filtered_words = [word for word in all_words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    

    def apply_text_cleaning(self, emotion_df, violence_df, hate_df):
        """Apply stopword removal to all datasets"""
        emotion_df['text'] = emotion_df['text'].apply(self.remove_stopwords)
        violence_df['text'] = violence_df['text'].apply(self.remove_stopwords)
        hate_df['text'] = hate_df['text'].apply(self.remove_stopwords)
        
        return emotion_df, violence_df, hate_df
    
    
    def process(self, emotion_df, violence_df, hate_df):
        """Complete preprocessing pipeline"""
        # Clean dataframes
        emotion_df, violence_df, hate_df = self.clean_dataframes(
            emotion_df, violence_df,hate_df
        )
        
        # Balance datasets
        emotion_df, violence_df, hate_df = self.balance_datasets(
            emotion_df, violence_df, hate_df
        )
        
        # Encode labels
        violence_df = self.encode_labels(violence_df)
        
        # Clean text
        emotion_df, violence_df, hate_df = self.apply_text_cleaning(
            emotion_df, violence_df, hate_df
        )
        
        return emotion_df, violence_df, hate_df



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

    print(emotion_df)